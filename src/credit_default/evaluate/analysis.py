"""Champion analysis: importance methods compared, error analysis, SHAP dependence plots and
the held-out predictions that the site's threshold/cost slider recomputes in the browser."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance

from credit_default.decision.engine import threshold_grid
from credit_default.evaluate.metrics import decision_metrics
from credit_default.features.build import engineer
from credit_default.models.bundle import ModelBundle

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- importance
def native_importance(bundle: ModelBundle, columns: list[str]) -> tuple[str, np.ndarray]:
    """The model's own importance measure on its input columns, and what it measures."""
    model = bundle.pipeline[-1]
    fam = bundle.family
    if fam == "lightgbm":
        return "split gain", model.booster_.feature_importance(importance_type="gain")
    if fam == "xgboost":
        score = model.get_booster().get_score(importance_type="total_gain")
        names = model.get_booster().feature_names or [f"f{i}" for i in range(len(columns))]
        return "total gain", np.array([score.get(n, 0.0) for n in names])
    if fam == "random_forest":
        return "impurity decrease", model.feature_importances_
    if fam == "catboost":
        return "prediction-values change", np.asarray(model.get_feature_importance())
    if fam == "logreg":
        return "|coefficient| (standardised inputs)", np.abs(model.coef_[0])
    raise ValueError(fam)


def importance_comparison(
    bundle: ModelBundle, X: pd.DataFrame, y, repeats: int, seed: int, sample: int = 1000
) -> dict:
    """Mean |SHAP| vs the model's native importance (model inputs), and permutation importance
    of the raw API inputs (PR-AUC drop on the test split)."""
    Xs = X.sample(min(sample, len(X)), random_state=seed)
    values, _, Xm = bundle.shap_values(Xs)
    shap_imp = np.abs(values).mean(axis=0)
    kind, native = native_importance(bundle, list(Xm.columns))
    native = np.asarray(native, dtype=float)
    rows = []
    if len(native) == len(Xm.columns):
        nat_share = native / native.sum() if native.sum() > 0 else native
        shap_share = shap_imp / shap_imp.sum()
        shap_rank = (-shap_imp).argsort().argsort() + 1
        nat_rank = (-native).argsort().argsort() + 1
        rows = [
            {
                "feature": str(c),
                "mean_abs_shap": float(shap_imp[i]),
                "shap_share": float(shap_share[i]),
                "shap_rank": int(shap_rank[i]),
                "native": float(native[i]),
                "native_share": float(nat_share[i]),
                "native_rank": int(nat_rank[i]),
            }
            for i, c in enumerate(Xm.columns)
        ]
        rho = float(spearmanr(shap_imp, native).statistic)
    else:  # one-hot LogReg: native importances are per encoded column
        rho = None
    rows.sort(key=lambda r: r["shap_rank"])
    perm = permutation_importance(
        bundle.pipeline,
        X,
        np.asarray(y),
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    perm_rows = sorted(
        (
            {
                "feature": c,
                "pr_auc_drop": float(perm.importances_mean[i]),
                "std": float(perm.importances_std[i]),
            }
            for i, c in enumerate(X.columns)
        ),
        key=lambda r: -r["pr_auc_drop"],
    )
    return {
        "native_kind": kind,
        "spearman_shap_vs_native": rho,
        "model_inputs": rows,
        "permutation_raw_inputs": perm_rows,
        "note": (
            "SHAP and native importances are associations inside the model, not causal effects. "
            "Permutation importance is measured on raw API inputs (the engineered ratios are "
            "recomputed from them), so excluded inputs score exactly 0."
        ),
    }


# ---------------------------------------------------------------------------- errors
def _segments(X: pd.DataFrame) -> dict[str, pd.Series]:
    f = engineer(X)
    pay0 = pd.cut(
        X["PAY_0"],
        [-3, -1, 0, 1, 9],
        labels=["paid duly / no use (-2, -1)", "revolving (0)", "1 month late", "2+ months late"],
    )
    util = pd.cut(
        f["util_1"].clip(-1, 10),
        [-np.inf, 0.1, 0.5, 0.9, np.inf],
        labels=["< 10%", "10-50%", "50-90%", ">= 90%"],
    )
    limit = pd.qcut(X["LIMIT_BAL"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
    return {"PAY_0 status": pay0, "utilisation (latest bill / limit)": util, "credit limit": limit}


def error_analysis(
    X: pd.DataFrame, y, proba, threshold: float, fn_cost: float, fp_cost: float
) -> dict:
    X = X.reset_index(drop=True)
    y = np.asarray(y).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    total_cost = decision_metrics(y, pred, fn_cost, fp_cost)["business_cost"] * len(y)
    segments = {}
    for name, seg in _segments(X).items():
        rows = []
        for value in seg.cat.categories:
            mask = (seg == value).to_numpy()
            if not mask.any():
                continue
            d = decision_metrics(y[mask], pred[mask], fn_cost, fp_cost)
            rows.append(
                {
                    "segment": str(value),
                    "n": int(mask.sum()),
                    "default_rate": float(y[mask].mean()),
                    "decline_rate": 1 - d["approval_rate"],
                    "fnr": d["fnr"],
                    "fpr": d["fpr"],
                    "cost_per_applicant": d["business_cost"],
                    "share_of_total_cost": d["business_cost"] * mask.sum() / total_cost
                    if total_cost
                    else 0.0,
                }
            )
        segments[name] = rows
    fn_mask, tp_mask = (y == 1) & (pred == 0), (y == 1) & (pred == 1)
    fp_mask, tn_mask = (y == 0) & (pred == 1), (y == 0) & (pred == 0)
    f = engineer(X)
    keys = ["LIMIT_BAL", "PAY_0", "max_delay", "util_1", "pay_ratio_1", "n_paid_duly"]
    profile = {
        name: {k: float(f.loc[m, k].mean()) if m.any() else None for k in keys}
        | {"n": int(m.sum())}
        for name, m in (("FN", fn_mask), ("TP", tp_mask), ("FP", fp_mask), ("TN", tn_mask))
    }
    return {
        "threshold": float(threshold),
        "segments": segments,
        "profiles": profile,
        "confident_misses": int((fn_mask & (p < threshold / 2)).sum()),
        "fn_share_pay0_not_late": float((X.loc[fn_mask, "PAY_0"] <= 0).mean())
        if fn_mask.any()
        else None,
        "note": "Segment rows are descriptive, not causal.",
    }


# ---------------------------------------------------------------------------- export
def export_predictions(y_val, p_val, y_test, p_test) -> dict:
    """Held-out predictions for in-browser recomputation, rounded only as far as every
    threshold on the decision grid still gives the same confusion matrix."""

    def same(y, p, q):
        return all(np.array_equal(p >= t, q >= t) for t in threshold_grid())

    out = {}
    for name, y, p in (("validation", y_val, p_val), ("test", y_test, p_test)):
        p = np.asarray(p, dtype=float)
        for digits in (5, 6, 8, 12):
            q = np.round(p, digits)
            if same(y, p, q):
                break
        out[name] = {
            "y": [int(v) for v in np.asarray(y)],
            "p": [float(v) for v in q],
            "digits": digits,
        }
    return out


def dependence_plots(
    bundle: ModelBundle, X: pd.DataFrame, out_dir: Path, top: list[str], seed: int
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    Xs = X.sample(min(1000, len(X)), random_state=seed)
    values, base, Xm = bundle.shap_values(Xs)
    exp = shap.Explanation(
        values=values,
        base_values=np.full(len(Xm), base),
        data=Xm.to_numpy(),
        feature_names=list(Xm.columns),
    )
    paths = {}
    for feat in top:
        if feat not in Xm.columns:
            continue
        plt.figure(figsize=(5.2, 3.6))
        shap.plots.scatter(exp[:, feat], show=False, color="#2563eb")
        plt.title(f"SHAP dependence: {feat} ({bundle.explanation_space})", fontsize=9)
        plt.tight_layout()
        path = out_dir / f"shap_dependence_{feat}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close("all")
        paths[f"shap_dependence_{feat}"] = path
    return paths


def waterfall_plot(bundle: ModelBundle, row: pd.DataFrame, out_path: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    values, base, Xm = bundle.shap_values(row)
    exp = shap.Explanation(
        values=values[0],
        base_values=base,
        data=Xm.iloc[0].to_numpy(),
        feature_names=list(Xm.columns),
    )
    plt.figure()
    shap.plots.waterfall(exp, max_display=10, show=False)
    plt.title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return out_path
