"""Global SHAP plots (bar summary and beeswarm) for the champion model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from credit_default.models.bundle import ModelBundle  # noqa: E402


def global_importance(bundle: ModelBundle, X: pd.DataFrame) -> pd.DataFrame:
    values, _, Xm = bundle.shap_values(X)
    imp = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": Xm.columns, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def global_shap_plots(
    bundle: ModelBundle, X: pd.DataFrame, out_dir: Path, sample: int, seed: int
) -> dict:
    import shap

    Xs = X.sample(min(sample, len(X)), random_state=seed)
    values, base, Xm = bundle.shap_values(Xs)
    explanation = shap.Explanation(
        values=values,
        base_values=np.full(len(Xm), base),
        data=Xm.to_numpy(),
        feature_names=list(Xm.columns),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, plot in (("shap_bar", shap.plots.bar), ("shap_beeswarm", shap.plots.beeswarm)):
        plt.figure()
        plot(explanation, max_display=15, show=False)
        plt.title(f"SHAP ({bundle.family}, log-odds before calibration)", fontsize=10)
        plt.tight_layout()
        path = out_dir / f"{name}.png"
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close("all")
        paths[name] = path
    imp = global_importance(bundle, Xs)
    imp.to_csv(out_dir / "shap_importance.csv", index=False)
    return paths
