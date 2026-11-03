"""The deployable unit: calibrated pipeline + decision threshold + explainer inputs + metadata.

The same joblib file is logged to MLflow (and registered) and committed as the local fallback,
so serving code has a single loading path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from credit_default.data.schema import FEATURE_COLUMNS

BUNDLE_FILENAME = "model.joblib"


@dataclass
class ModelBundle:
    family: str
    pipeline: Any  # fitted base pipeline (features -> preprocess -> model)
    calibrator: Any  # CalibratedClassifierCV(FrozenEstimator(pipeline), "isotonic")
    threshold: float  # cost-optimal threshold on calibrated probabilities
    background: pd.DataFrame  # small raw sample, used by linear SHAP
    metadata: dict = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    # ---- prediction -------------------------------------------------------------------------
    def _frame(self, X) -> pd.DataFrame:
        df = pd.DataFrame(X)
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        return df[self.feature_columns]

    def predict_proba(self, X) -> np.ndarray:
        """Calibrated probability of default."""
        return self.calibrator.predict_proba(self._frame(X))[:, 1]

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ---- explanation ------------------------------------------------------------------------
    def _model_input(self, X) -> pd.DataFrame:
        out = self._frame(X)
        for _, step in self.pipeline.steps[:-1]:
            if step is None or step == "passthrough" or hasattr(step, "fit_resample"):
                continue  # samplers (SMOTE) only act at fit time
            out = step.transform(out)
        return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)

    def explainer(self):
        if getattr(self, "_explainer", None) is None:
            import shap

            model = self.pipeline[-1]
            if self.family == "logreg":
                self._explainer = shap.LinearExplainer(model, self._model_input(self.background))
            else:
                self._explainer = shap.TreeExplainer(model)
        return self._explainer

    def shap_values(self, X) -> tuple[np.ndarray, float, pd.DataFrame]:
        """SHAP values in the base model's log-odds space (before isotonic calibration)."""
        Xm = self._model_input(X)
        exp = self.explainer()
        values = exp.shap_values(Xm)
        if isinstance(values, list):  # older shap returns one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 1]
        base = exp.expected_value
        base = float(np.ravel(base)[-1])
        return values, base, Xm

    def explain(self, X, top_k: int = 5) -> list[dict]:
        values, base, Xm = self.shap_values(X)
        probs = self.predict_proba(X)
        results = []
        for i in range(len(Xm)):
            order = np.argsort(-np.abs(values[i]))[:top_k]
            results.append(
                {
                    "probability": float(probs[i]),
                    "base_value": base,
                    "contributions": [
                        {
                            "feature": str(Xm.columns[j]),
                            "value": float(Xm.iloc[i, j]),
                            "shap_value": float(values[i, j]),
                        }
                        for j in order
                    ],
                }
            )
        return results

    # ---- persistence ------------------------------------------------------------------------
    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_explainer", None)  # rebuilt lazily; shap explainers do not pickle reliably
        return state

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".joblib":
            path = path / BUNDLE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: str | Path) -> ModelBundle:
        path = Path(path)
        if path.is_dir():
            path = path / BUNDLE_FILENAME
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return bundle
