# Architecture

The system is one Python package (`src/credit_default`), Hydra configs (`configs/`), a Prefect
flow, six GitHub workflows and a static dashboard. Nothing runs permanently except the API on
Render; everything else is a CI job whose outputs are committed to `results/`.

## Lifecycle

1. **Ingest and version** (`data/download.py`, `dvc.yaml`): download, normalise to one canonical
   CSV, verify sha256, DVC-track.
2. **Validate** (`validation/schema.py`): Pandera contract; a violation stops the pipeline.
3. **Describe** (`data/quality.py`): data-quality and leakage report (`results/data_quality.md`).
4. **Train** (`pipeline/train.py`): every family under one protocol (see [modelling](modelling.md)),
   champion selection, calibration, cutoff, analysis files, MLflow logging, `@staging`.
5. **Gate** (`registry/gate.py` + `registry/policy.py`): 12 checks against `@production` on the
   same test split; pass moves `@production` (old version kept as `@previous-production`).
6. **Record** (`registry/governance.py record`): append the decision to
   `results/governance/history.json`, rejections included.
7. **Publish** (`pipeline/publish.py`): pass → `results/` + `models/bundled/`; fail →
   `results/challenger/` only.
8. **Serve** (`serve/`): Render builds from `main` and serves `models/bundled/model.joblib`.
9. **Monitor** (`monitor/`): drift scenarios and the policy; `retrain.yml` calls `train.yml`
   only when the policy says `retrain`.
10. **Report** (`scripts/build_site.py`): the dashboard reads only committed files.

## Where the durable state lives

| State | Location | Why |
|---|---|---|
| Data version | `dvc.lock` + pinned sha256 | reproducible, no secrets |
| Served model | `models/bundled/` in git | Render has no access to a private registry; git history doubles as the model archive for rollback |
| Promotion decisions | `results/governance/history.json` | CI's MLflow store is per run |
| Metrics and analyses | `results/*.json` | every number on the site and in the README comes from here |
| MLflow runs and aliases | per-run SQLite store (uploaded as a CI artifact); docker-compose runs a persistent server | aliases are the registry contract; the bootstrap step re-creates `@production` from the committed bundle |

## Configuration

`configs/config.yaml` composes data, train, cost, monitor and registry groups. Every threshold
the system acts on is configuration: cost ratio and policy constraints (`cost/`), promotion
checks (`registry/default.yaml` → `gate`), monitoring policy (`monitor/default.yaml` → `policy`),
excluded inputs (`features.exclude`). API settings are environment variables (`.env.example`).

## Workflows

See the CI/CD table in the README. Heavy compute (Optuna over five families, the benchmark,
Evidently reports, Playwright) runs on GitHub-hosted runners; results come back as bot commits
with a `git pull --rebase` retry loop.
