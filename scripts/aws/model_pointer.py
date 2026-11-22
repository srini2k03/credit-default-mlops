"""Write the JSON pointer published next to the production model in S3.

The ECS task loads ``s3://<bucket>/models/production/model.joblib`` (MODEL_SOURCE=s3); this
pointer (``models/production/model.json``) records which registry version and commit it is.

Usage: python scripts/aws/model_pointer.py <bundle path> <s3 uri of the bundle> > model.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path


def registry_version() -> str | None:
    try:
        from credit_default.config import load_config
        from credit_default.registry import mlflow_registry as reg

        cfg = load_config([])
        reg.setup_tracking(cfg)
        mv = reg.get_version(cfg, cfg.registry.production_alias)
        return str(mv.version) if mv is not None else None
    except Exception as exc:  # the pointer is informational; never fail the publish on it
        print(f"registry version unavailable: {exc}", file=sys.stderr)
        return None


def main(bundle_path: str, s3_uri: str) -> dict:
    from credit_default.models.bundle import ModelBundle

    path = Path(bundle_path)
    bundle = ModelBundle.load(path)
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo, run_id = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_RUN_ID")
    return {
        "s3_uri": s3_uri,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "registry_model": "models:/credit-default@production",
        "registry_version": registry_version(),
        "bundle_git_commit": bundle.metadata.get("git_commit"),
        "family": bundle.family,
        "threshold": bundle.threshold,
        "published_commit": os.environ.get("GITHUB_SHA"),
        "published_by_run": f"{server}/{repo}/actions/runs/{run_id}" if repo and run_id else None,
        "published_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    print(json.dumps(main(sys.argv[1], sys.argv[2]), indent=2))
