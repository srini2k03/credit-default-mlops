"""Download the UCI table (UCI archive first, pinned GitHub mirrors as fallback) and verify it.

Every source is converted to one canonical CSV (UCI column names, integer values, target renamed
to ``default_payment_next_month``). Two checks guard the data:

1. the downloaded file's sha256 against the pinned value for that source (when pinned), and
2. the canonical CSV's sha256, which is identical whichever source succeeded.

Usage:  python -m credit_default.data.download [data=uci] [hydra overrides...]
"""

from __future__ import annotations

import hashlib
import io
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import requests

from credit_default.config import REPO_ROOT, load_config, split_cli
from credit_default.data.schema import ID_COLUMN, RAW_COLUMNS, TARGET, UCI_TARGET_NAMES

log = logging.getLogger(__name__)


class ChecksumError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check(data: bytes, expected: str | None, what: str) -> None:
    if expected is None:
        return
    got = sha256_bytes(data)
    if got != expected:
        raise ChecksumError(f"{what}: sha256 {got} != expected {expected}")


def _fetch(url: str, timeout: float) -> bytes:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "credit-default-mlops"})
    resp.raise_for_status()
    data = resp.content
    if data.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"{url} returned a Git LFS pointer, not the file")
    return data


def _read_xls(data: bytes) -> pd.DataFrame:
    # Row 0 holds X1..X23/Y codes; row 1 holds the real column names.
    return pd.read_excel(io.BytesIO(data), header=1, engine="xlrd")


def to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise any source layout to the canonical column names and integer dtypes."""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    for name in UCI_TARGET_NAMES:
        if name in df.columns:
            df = df.rename(columns={name: TARGET})
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"source is missing columns: {missing}")
    out = df[RAW_COLUMNS].copy()
    for col in RAW_COLUMNS:
        values = pd.to_numeric(out[col], errors="raise")
        if (values.round() != values).any():
            raise ValueError(f"column {col} has non-integer values")
        out[col] = values.round().astype("int64")
    return out.sort_values(ID_COLUMN).reset_index(drop=True)


def write_canonical(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return sha256_file(path)


def load_source(source: dict, timeout: float) -> pd.DataFrame:
    data = _fetch(source["url"], timeout)
    kind = source["kind"]
    _check(data, source.get("sha256"), source["name"])
    if kind == "uci_zip":
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            xls_names = [n for n in zf.namelist() if n.lower().endswith(".xls")]
            if not xls_names:
                raise RuntimeError("zip has no .xls member")
            inner = zf.read(xls_names[0])
        try:
            _check(inner, source.get("inner_sha256"), f"{source['name']} (inner .xls)")
        except ChecksumError as exc:  # UCI may re-pack the file; the canonical sha decides
            log.warning("%s", exc)
        return _read_xls(inner)
    if kind == "uci_xls":
        return _read_xls(data)
    if kind == "kaggle_csv":
        return pd.read_csv(io.BytesIO(data))
    raise ValueError(f"unknown source kind {kind!r}")


def download(cfg) -> Path:
    """Try each configured source in order; write and verify the canonical CSV."""
    out_path = REPO_ROOT / cfg.data.raw_path
    expected = cfg.data.canonical_sha256
    if out_path.exists() and sha256_file(out_path) == expected:
        log.info("%s already present and verified", out_path)
        return out_path

    errors = []
    for source in cfg.data.sources:
        source = dict(source)
        try:
            log.info("downloading from %s: %s", source["name"], source["url"])
            df = to_canonical(load_source(source, cfg.data.timeout_s))
            if len(df) != cfg.data.expected_rows:
                raise ValueError(f"expected {cfg.data.expected_rows} rows, got {len(df)}")
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp) / "canonical.csv"
                digest = write_canonical(df, tmp_path)
                if expected and digest != expected:
                    raise ChecksumError(f"canonical CSV sha256 {digest} != expected {expected}")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.replace(out_path)
            log.info(
                "wrote %s (%d rows, sha256 %s) from %s", out_path, len(df), digest, source["name"]
            )
            return out_path
        except Exception as exc:  # try the next source, but remember why this one failed
            log.warning("source %s failed: %s", source["name"], exc)
            errors.append(f"{source['name']}: {exc}")
    raise RuntimeError("all data sources failed:\n  " + "\n  ".join(errors))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    overrides, _ = split_cli(sys.argv[1:] if argv is None else argv)
    cfg = load_config(overrides)
    if cfg.data.source == "synthetic":
        from credit_default.data.synthetic import write_fixture

        write_fixture(cfg)
        return 0
    download(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
