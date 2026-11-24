"""Fast scoring path for the API: the same model and the same arithmetic, without the pandas
overhead of the training pipeline on one-row requests.

``ModelBundle.predict_proba`` runs the sklearn pipeline: build a DataFrame, engineer features
with pandas, predict, calibrate. For a single request most of the ~6 ms is pandas bookkeeping
in feature engineering. ``FastScorer`` engineers features on a NumPy array with the same
float64 operations in the same order, calls the same fitted model, and applies the bundle's
own fitted calibrator object.

Safety: the scorer is only enabled if it reproduces ``ModelBundle.predict_proba`` exactly
(``np.array_equal``, not approximately) on a verification sample at load time; otherwise the
API uses the pipeline. Tests and the benchmark assert the same on the real test split.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from credit_default.data.schema import BILL_AMT, FEATURE_COLUMNS, PAY_AMT, PAY_STATUS
from credit_default.features.build import ENGINEERED_COLUMNS, FeatureBuilder

log = logging.getLogger(__name__)

_IDX = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
_BILL = [_IDX[c] for c in BILL_AMT]
_PAY = [_IDX[c] for c in PAY_AMT]
_STATUS = [_IDX[c] for c in PAY_STATUS]


def engineer_array(X: np.ndarray) -> np.ndarray:
    """NumPy twin of ``features.build.engineer``: raw columns (FEATURE_COLUMNS order) plus the
    engineered columns, float64, identical values."""
    out = X.astype(np.float64, copy=True)
    edu, mar = _IDX["EDUCATION"], _IDX["MARRIAGE"]
    out[:, edu] = np.where((out[:, edu] >= 1) & (out[:, edu] <= 4), out[:, edu], 4.0)
    out[:, mar] = np.where((out[:, mar] >= 1) & (out[:, mar] <= 3), out[:, mar], 3.0)
    limit = np.clip(out[:, _IDX["LIMIT_BAL"]], 1.0, None)
    bills, pays, status = out[:, _BILL], out[:, _PAY], out[:, _STATUS]
    eng = np.empty((len(out), len(ENGINEERED_COLUMNS)))
    eng[:, 0] = bills[:, 0] / limit
    eng[:, 1] = bills.mean(axis=1) / limit
    eng[:, 2] = (bills[:, 0] - bills[:, 5]) / limit
    eng[:, 3] = np.clip(pays[:, 0] / np.maximum(bills[:, 1], 1.0), 0, 10)
    eng[:, 4] = np.clip(
        pays.sum(axis=1) / np.maximum(np.clip(bills, 0, None).sum(axis=1), 1.0), 0, 10
    )
    eng[:, 5] = status.max(axis=1)
    eng[:, 6] = (status > 0).sum(axis=1)
    eng[:, 7] = (status == -1).sum(axis=1)
    return np.hstack([out, eng])


class FastScorer:
    """Scores rows given as dicts or arrays; built from a bundle, enabled only if exact."""

    def __init__(self, bundle):
        steps = dict(bundle.pipeline.steps)
        if not isinstance(steps.get("features"), FeatureBuilder) or steps.get("preprocess") not in (
            "passthrough",
            None,
        ):
            raise ValueError(f"no fast path for family {bundle.family!r} (stateful preprocessing)")
        drop = set(getattr(steps["features"], "drop", ()))
        names = [c for c in FEATURE_COLUMNS + ENGINEERED_COLUMNS if c not in drop]
        self._keep = [
            i for i, c in enumerate(FEATURE_COLUMNS + ENGINEERED_COLUMNS) if c not in drop
        ]
        self._names = names
        self._model = bundle.pipeline[-1]
        self._calibrator = None
        if bundle.calibrator is not bundle.pipeline:
            ccs = bundle.calibrator.calibrated_classifiers_
            if len(ccs) != 1 or len(ccs[0].calibrators) != 1:
                raise ValueError("expected one fitted binary calibrator")
            self._calibrator = ccs[0].calibrators[0]
        self.threshold = bundle.threshold

    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(engineer_array(X)[:, self._keep], columns=self._names)

    def raw(self, X: np.ndarray) -> np.ndarray:
        """Uncalibrated probability (what ``bundle.pipeline.predict_proba`` returns)."""
        return self._model.predict_proba(self._frame(X))[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self._calibrator is None:
            return self.raw(X)
        # sklearn's calibrators consume decision_function when the model has one (the raw
        # margin for boosted trees), else the positive-class probability.
        frame = self._frame(X)
        if hasattr(self._model, "decision_function"):
            score = np.asarray(self._model.decision_function(frame), dtype=np.float64)
        else:
            score = self._model.predict_proba(frame)[:, 1]
        p = self._calibrator.predict(score.ravel())
        # Same post-processing as sklearn's _CalibratedClassifier for binary problems.
        p = np.asarray(p, dtype=np.float64)
        p[(p > 1.0) & (p <= 1.0 + 1e-5)] = 1.0
        return p

    @staticmethod
    def rows_to_array(rows: list[dict]) -> np.ndarray:
        return np.array([[r[c] for c in FEATURE_COLUMNS] for r in rows], dtype=np.float64)


def build_fast_scorer(bundle, verify: pd.DataFrame) -> FastScorer | None:
    """A FastScorer if it reproduces the bundle exactly on ``verify``; otherwise None."""
    try:
        scorer = FastScorer(bundle)
        expected = bundle.predict_proba(verify)
        got = scorer.predict_proba(verify[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    except Exception as exc:  # any surprise: fall back to the pipeline
        log.info("fast scoring disabled: %s", exc)
        return None
    if not np.array_equal(expected, got):
        log.warning(
            "fast scoring disabled: max abs difference %.3g on %d rows",
            float(np.max(np.abs(expected - got))),
            len(verify),
        )
        return None
    log.info("fast scoring enabled (bit-identical on %d verification rows)", len(verify))
    return scorer
