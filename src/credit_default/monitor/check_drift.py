"""Scheduled drift check. Exit 0 = no action / monitor, 1 = retrain, 2 = error, 3 = blocked.

The decision comes from the monitoring policy (``configs/monitor/default.yaml`` -> ``policy``,
see ``credit_default.monitor.scenarios``): data-quality failures block the batch, feature or
prediction drift beyond the limits starts a (gated) challenger, smaller movements are only
monitored. Labels are not available at check time, so the label-free branch of the policy is
used. An Evidently report is still written for every check, for people to read.

Usage:
  python -m credit_default.monitor.check_drift --slice=drifted     # simulated covariate shift
  python -m credit_default.monitor.check_drift --slice=control     # undrifted control slice
  python -m credit_default.monitor.check_drift --slice=missing_values   # any scenario name
  python -m credit_default.monitor.check_drift --slice=log --log-path=logs/predictions.sqlite

Writes ``drift``, ``action`` and ``drift_share`` to $GITHUB_OUTPUT when it is set.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.data.schema import FEATURE_COLUMNS

log = logging.getLogger(__name__)
EXIT = {"none": 0, "monitor": 0, "retrain": 1, "block": 3}


def _flag(flags, name, default=None):
    return next((f.split("=", 1)[1] for f in flags if f.startswith(f"--{name}=")), default)


def assess(current, reference, bundle, cfg) -> dict:
    """Label-free policy decision for one batch of applications."""
    from credit_default.data.split import make_splits
    from credit_default.monitor import scenarios

    pol = cfg.monitor.policy
    ref_X, X = reference[FEATURE_COLUMNS], current[FEATURE_COLUMNS]
    dq = scenarios.data_quality(current)
    fdrift = scenarios.feature_drift(ref_X, X, pol)
    pdrift = {"psi": float("nan")}
    if bundle is not None:
        df = None
        try:
            import pandas as pd

            from credit_default.validation.schema import validate

            df = validate(pd.read_csv(REPO_ROOT / cfg.data.raw_path), cfg)
        except Exception as exc:  # the reference scores need the labelled data
            log.warning("no held-out reference scores: %s", exc)
        if df is not None:
            splits = make_splits(df, cfg.train.splits, cfg.seed)
            p_ref = bundle.predict_proba(splits.validation[FEATURE_COLUMNS])
            try:
                p_cur = bundle.predict_proba(X)
            except ValueError:
                p_cur = bundle.predict_proba(X.fillna(ref_X.median()))
            pdrift = scenarios.prediction_drift(p_ref, p_cur, bundle.threshold)
    labels = {"n": 0, "p_value": 1.0, "rate": None}
    decision = scenarios.decide(dq, fdrift, pdrift, None, labels, pol)
    return {
        "data_quality": dq,
        "feature_drift_share": float(np.mean([r["drifted"] for r in fdrift])),
        "drifted_features": [r["feature"] for r in fdrift if r["drifted"]],
        "prediction_psi": pdrift["psi"],
        "decision": decision,
    }


def main(argv: list[str] | None = None) -> int:
    import pandas as pd

    from credit_default.monitor.drift import load_bundle_if_any, run_report, with_prediction
    from credit_default.monitor.scenarios import SCENARIOS, make_scenario
    from credit_default.monitor.simulate import build_slices
    from credit_default.pipeline.steps import write_json
    from credit_default.validation.schema import validate

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, flags = split_cli(sys.argv[1:] if argv is None else argv)
    cfg = load_config(overrides)
    name = _flag(flags, "slice", "drifted")
    out_dir = REPO_ROOT / cfg.paths.reports_dir
    try:
        slices = build_slices(cfg)
        if name == "log":
            from credit_default.serve.prediction_log import read_log

            logged = read_log(_flag(flags, "log-path", REPO_ROOT / cfg.paths.prediction_log))
            if len(logged) < 100:
                log.warning("only %d logged predictions; too few to judge drift", len(logged))
                return 0
            current = logged[FEATURE_COLUMNS]
        elif name in slices:
            current = slices[name]
        elif name in SCENARIOS:
            from credit_default.data.split import make_splits

            df = validate(pd.read_csv(REPO_ROOT / cfg.data.raw_path), cfg)
            pool = make_splits(df, cfg.train.splits, cfg.seed).test.reset_index(drop=True)
            current = make_scenario(name, pool, cfg.monitor.slice_rows, cfg, cfg.monitor.slice_seed)
            current = current[FEATURE_COLUMNS]
        else:
            raise ValueError(f"unknown slice {name!r}")
        bundle = load_bundle_if_any(flags, cfg)
        result = assess(current, slices["reference"], bundle, cfg)
        evidently = None
        if not current.isna().any().any():  # Evidently's report needs complete columns
            evidently = run_report(
                with_prediction(slices["reference"], bundle),
                with_prediction(current, bundle),
                name,
                out_dir,
                cfg.monitor.drift_share_threshold,
            )
        result.update(
            {"slice": name, "evidently_drift_share": evidently and evidently["drift_share"]}
        )
        write_json(result, out_dir / f"check_{name}.json")
    except Exception as exc:
        log.exception("drift check failed: %s", exc)
        return 2

    action = result["decision"]["action"]
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.write(
                f"drift={'true' if action == 'retrain' else 'false'}\naction={action}\n"
                f"drift_share={result['feature_drift_share']:.4f}\n"
            )
    reasons = "; ".join(result["decision"]["reasons"]) or "nothing moved"
    log.log(
        logging.WARNING if action in {"retrain", "block"} else logging.INFO,
        "slice %s -> %s (%s)",
        name,
        action.upper(),
        reasons,
    )
    return EXIT[action]


if __name__ == "__main__":
    raise SystemExit(main())
