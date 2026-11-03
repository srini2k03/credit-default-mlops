"""Hydra config loading via the compose API (no working-directory changes, no output dirs)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.environ.get("CREDIT_DEFAULT_CONFIG_DIR", REPO_ROOT / "configs"))


def load_config(overrides: Sequence[str] = ()) -> DictConfig:
    """Compose the root config with Hydra override syntax (``key=value``, ``group=option``)."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base="1.3"):
        cfg = compose(config_name="config", overrides=list(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def split_cli(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split argv into Hydra overrides (``a=b``) and plain flags (``--flag``)."""
    overrides = [a for a in argv if "=" in a and not a.startswith("--")]
    flags = [a for a in argv if a not in overrides]
    return overrides, flags
