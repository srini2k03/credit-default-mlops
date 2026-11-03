"""Static figures for the run report (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402

from credit_default.threshold.cost import CostCurve  # noqa: E402

INK = "#1f2937"
ACCENT = "#2563eb"
MUTED = "#9ca3af"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_cost_curve(curve: CostCurve, markers: dict[str, float], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(curve.thresholds, curve.costs, color=ACCENT, lw=2, label="expected cost / applicant")
    colors = {"cost-optimal": "#059669", "0.5": "#dc2626", "max-F1": "#d97706"}
    for name, thr in markers.items():
        idx = int(np.argmin(np.abs(curve.thresholds - thr)))
        ax.scatter(
            [curve.thresholds[idx]],
            [curve.costs[idx]],
            s=60,
            zorder=3,
            color=colors.get(name, INK),
            label=f"{name}: t={thr:.2f}, cost={curve.costs[idx]:.3f}",
        )
    ax.set_xlabel("decision threshold on calibrated P(default)")
    ax.set_ylabel("expected cost per applicant (FP units)")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_calibration(y_true, probs: dict[str, np.ndarray], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
    for name, p in probs.items():
        frac, mean = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", lw=1.5, label=name)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed default rate")
    ax.set_title("Reliability on the test holdout", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
