"""Drift gate for scheduled monitoring. Exit 0 = no drift, 1 = drift above threshold, 2 = error.

Usage:
  python -m credit_default.monitor.check_drift --slice=drifted     # simulated production slice
  python -m credit_default.monitor.check_drift --slice=control     # undrifted control slice
  python -m credit_default.monitor.check_drift --slice=log --log-path=logs/predictions.sqlite

Writes ``drift=true|false`` and ``drift_share=...`` to $GITHUB_OUTPUT when it is set.
"""

from __future__ import annotations

import logging
import os
import sys

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.data.schema import FEATURE_COLUMNS

log = logging.getLogger(__name__)


def _flag(flags, name, default=None):
    return next((f.split("=", 1)[1] for f in flags if f.startswith(f"--{name}=")), default)


def main(argv: list[str] | None = None) -> int:
    from credit_default.monitor.drift import load_bundle_if_any, run_report, with_prediction
    from credit_default.monitor.simulate import build_slices

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, flags = split_cli(sys.argv[1:] if argv is None else argv)
    cfg = load_config(overrides)
    name = _flag(flags, "slice", "drifted")
    threshold = float(_flag(flags, "threshold", cfg.monitor.drift_share_threshold))
    try:
        slices = build_slices(cfg)
        if name == "log":
            from credit_default.serve.prediction_log import read_log

            logged = read_log(_flag(flags, "log-path", REPO_ROOT / cfg.paths.prediction_log))
            if len(logged) < 100:
                log.warning("only %d logged predictions; too few to judge drift", len(logged))
                return 0
            current = logged[FEATURE_COLUMNS]
        else:
            current = slices[name]
        bundle = load_bundle_if_any(flags, cfg)
        summary = run_report(
            with_prediction(slices["reference"], bundle),
            with_prediction(current, bundle),
            name,
            REPO_ROOT / cfg.paths.reports_dir,
            threshold,
        )
    except Exception as exc:
        log.exception("drift check failed: %s", exc)
        return 2

    drift = summary["drift_detected"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(
                f"drift={'true' if drift else 'false'}\ndrift_share={summary['drift_share']:.4f}\n"
            )
    if drift:
        log.warning("DRIFT: share %.2f > %.2f on slice %s", summary["drift_share"], threshold, name)
        return 1
    log.info("no drift: share %.2f <= %.2f on slice %s", summary["drift_share"], threshold, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
