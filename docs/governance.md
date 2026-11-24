# Champion/challenger governance

A new model never replaces the champion because it was trained; it replaces it only by passing
every check of the promotion policy, and every decision (pass or fail) is recorded.

## Registry contract

MLflow aliases replace the deprecated stages: every training run registers its champion as
`@staging`; the gate moves `@production` on a pass and keeps the previous version as
`@previous-production`. The API resolves `models:/credit-default@production` (or, on Render,
the committed bundle that the same gate produced).

## Promotion policy (`configs/registry/default.yaml` → `gate`)

| Group | Check | Limit |
|---|---|---|
| performance | ROC-AUC floor | ≥ 0.70 |
| performance | PR-AUC floor | ≥ 0.45 |
| performance | PR-AUC vs champion | ≥ champion − 0.005 |
| calibration | ECE ceiling | ≤ 0.05 |
| calibration | Brier vs champion | ≤ champion + 0.002 |
| cost | expected cost vs champion | ≤ champion × 1.01 |
| operational | API input contract | columns unchanged |
| operational | probabilities valid | finite, in [0, 1] |
| operational | deterministic | same input → same output |
| operational | cutoff valid | in (0, 1) |
| operational | latency | single-row in-process p95 ≤ 50 ms |
| operational | bundle size | ≤ 50 MB |

Both models are re-scored on the same test split at their own cutoffs. All checks are
evaluated (no short-circuit) and written to `gate.json` / `gate.md`. With no champion the
comparative checks are skipped and the absolute ones still apply.

## What happens in CI (`train.yml`)

- gate passes → bundle, results and README table are published; the API image is load-tested;
  Render redeploys from `main`.
- gate fails → the job still succeeds, but the challenger's report goes to `results/challenger/`,
  a warning is raised, `models/bundled/` is untouched, and the rejection is appended to
  `results/governance/history.json` with the failed checks.

Real decisions so far (`results/governance/history.json`):

1. train run #1 (Nov 3): first version accepted (backfilled from that run's gate.json).
2. train run 45371941736: the LightGBM/Platt challenger passed all 12 checks against it
   (PR-AUC 0.5594 vs 0.5362, Brier 0.1346 vs 0.1353, cost 0.5578 vs 0.5625, ECE 0.0146,
   p95 6.6 ms, 0.3 MB).
3. retrain run 45379624037, triggered by the monitoring policy on the covariate-shift batch:
   the retrained challenger was identical to production on every metric (the retrain re-uses
   the same labelled data) and passed.

## Rollback

- Registry: `python -m credit_default.registry.governance rollback --mode=registry` swaps
  `@production` and `@previous-production` and records the event.
- Served bundle (Render): `rollback --mode=git [--to=<sha>]` restores `models/bundled/` from the
  previous bundle commit (found with `git log`), records the event; committing and pushing
  redeploys.

## Simulation (`results/governance/demo.md`)

Run by `train.yml` in a throw-away MLflow registry on the real splits, so both outcomes are
demonstrated without touching the served model:

| Candidate | What it is | Outcome | Failed checks |
|---|---|---|---|
| v1 baseline | default hyper-parameters, 50% of training rows | promoted (first version) | - |
| v2 retrained | tuned, all training rows | promoted (PR-AUC 0.527 → 0.560) | - |
| v3 noisy labels | outcome feed recorded 40% of defaults as repaid | **rejected**, v2 stays | PR-AUC, Brier and cost vs champion |
| v4 uncalibrated | v2's base model served without a calibrator | **rejected**, v2 stays | ECE (0.190), Brier vs champion |
| rollback | simulated incident after v2 went live | `@production` v2 → v1 | - |

v4 is the instructive case: its expected cost (0.5553) is slightly *lower* than v2's (0.5562)
because the cutoff re-chosen on validation absorbs the inflated scores, yet its probabilities
are off by 0.19 on average. A cost-only gate would have promoted it; the API returns those
probabilities to callers, so calibration is a promotion criterion in its own right.

Tests (`tests/test_governance.py`) prove each check fires, that a failed challenger cannot move
`@production` in registry mode, that promotion and rollback move the aliases as described, that
the history is append-only and records rejections, and that git rollback restores the previous
bundle.
