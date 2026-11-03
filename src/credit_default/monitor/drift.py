"""Evidently data-drift reports (HTML + JSON summary) for the control and drifted slices.

Usage: python -m credit_default.monitor.drift [overrides] [--bundle=models/bundled/model.joblib]
If a model bundle is available, its predicted probability is added as a monitored column
(prediction drift).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.pipeline.steps import now_iso, write_json

log = logging.getLogger(__name__)


def with_prediction(df: pd.DataFrame, bundle) -> pd.DataFrame:
    if bundle is None:
        return df
    out = df.copy()
    out["prediction"] = bundle.predict_proba(df)
    return out


def drift_summary(reference: pd.DataFrame, current: pd.DataFrame) -> tuple[dict, object]:
    """Run Evidently's DataDriftPreset; return (summary, snapshot)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
    result = snapshot.dict()
    share, count, columns = None, None, {}
    for m in result["metrics"]:
        name = m["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            share, count = float(m["value"]["share"]), int(m["value"]["count"])
        elif name.startswith("ValueDrift"):
            cfg = m["config"]
            columns[cfg["column"]] = {
                "score": float(m["value"]),
                "method": cfg.get("method"),
                "threshold": cfg.get("threshold"),
            }
    if share is None:
        raise RuntimeError("Evidently result has no DriftedColumnsCount metric")
    return {
        "drift_share": share,
        "drifted_columns": count,
        "n_columns": len(current.columns),
        "n_reference": len(reference),
        "n_current": len(current),
        "columns": columns,
    }, snapshot


def run_report(reference, current, name: str, out_dir: Path, threshold: float) -> dict:
    summary, snapshot = drift_summary(reference, current)
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / f"drift_{name}.html"
    snapshot.save_html(str(html))
    summary.update(
        {
            "slice": name,
            "threshold": threshold,
            "drift_detected": summary["drift_share"] > threshold,
            "generated_at": now_iso(),
            "html": html.name,
        }
    )
    write_json(summary, out_dir / f"drift_{name}.json")
    log.info(
        "%s: %d/%d columns drifted (share %.2f, threshold %.2f) -> %s",
        name,
        summary["drifted_columns"],
        summary["n_columns"],
        summary["drift_share"],
        threshold,
        html,
    )
    return summary


def load_bundle_if_any(flags, cfg):
    from credit_default.models.bundle import ModelBundle

    path = next((f.split("=", 1)[1] for f in flags if f.startswith("--bundle=")), None)
    path = Path(path) if path else REPO_ROOT / cfg.paths.bundle_dir / "model.joblib"
    if path.exists():
        return ModelBundle.load(path)
    log.info("no model bundle at %s: reporting feature drift only", path)
    return None


def main(argv: list[str] | None = None) -> int:
    from credit_default.monitor.simulate import build_slices, write_slices

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, flags = split_cli(sys.argv[1:] if argv is None else argv)
    cfg = load_config(overrides)
    slices = build_slices(cfg)
    write_slices(cfg, slices)
    bundle = load_bundle_if_any(flags, cfg)
    ref = with_prediction(slices["reference"], bundle)
    out_dir = REPO_ROOT / cfg.paths.reports_dir
    summaries = [
        run_report(
            ref,
            with_prediction(slices[name], bundle),
            name,
            out_dir,
            cfg.monitor.drift_share_threshold,
        )
        for name in ("control", "drifted")
    ]
    write_json(
        {"generated_at": now_iso(), "data_source": cfg.data.source, "reports": summaries},
        out_dir / "drift_summary.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
