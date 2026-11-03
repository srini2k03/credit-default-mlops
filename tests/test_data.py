import io
import zipfile

import pandas as pd
import pytest

from credit_default.data import download as dl
from credit_default.data.schema import FEATURE_COLUMNS, RAW_COLUMNS, TARGET
from credit_default.data.synthetic import generate


def test_fixture_has_exact_uci_schema(small_df):
    assert list(small_df.columns) == RAW_COLUMNS
    assert len(FEATURE_COLUMNS) == 23
    assert all(str(t) == "int64" for t in small_df.dtypes)


def test_fixture_is_deterministic_and_has_uci_default_rate():
    a, b = generate(4000, seed=1), generate(4000, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert 0.19 <= a[TARGET].mean() <= 0.25
    assert not a.equals(generate(4000, seed=2))


def test_canonicalises_kaggle_csv_layout(small_df):
    kaggle = small_df.rename(columns={TARGET: "default.payment.next.month"}).astype(float)
    out = dl.to_canonical(kaggle)
    assert list(out.columns) == RAW_COLUMNS
    pd.testing.assert_frame_equal(out, small_df.sort_values("ID").reset_index(drop=True))


def test_canonical_rejects_missing_columns(small_df):
    with pytest.raises(ValueError, match="missing columns"):
        dl.to_canonical(small_df.drop(columns=["PAY_0"]))


def _cfg_with(sources, canonical_sha, raw_path, rows):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "data": {
                "raw_path": str(raw_path),
                "canonical_sha256": canonical_sha,
                "expected_rows": rows,
                "timeout_s": 5,
                "sources": sources,
            }
        }
    )


def test_download_falls_back_and_verifies_checksum(tmp_path, small_df, monkeypatch):
    csv_bytes = (
        small_df.rename(columns={TARGET: "default.payment.next.month"}).to_csv(index=False).encode()
    )
    expected = dl.write_canonical(dl.to_canonical(small_df), tmp_path / "ref.csv")
    calls = []

    def fake_fetch(url, timeout):
        calls.append(url)
        if "primary" in url:
            raise ConnectionError("blocked")
        return csv_bytes

    monkeypatch.setattr(dl, "_fetch", fake_fetch)
    sources = [
        {"name": "primary", "url": "https://primary/x.zip", "kind": "uci_zip"},
        {
            "name": "mirror",
            "url": "https://mirror/x.csv",
            "kind": "kaggle_csv",
            "sha256": dl.sha256_bytes(csv_bytes),
        },
    ]
    out = dl.download(_cfg_with(sources, expected, tmp_path / "raw.csv", len(small_df)))
    assert calls == ["https://primary/x.zip", "https://mirror/x.csv"]
    assert dl.sha256_file(out) == expected


def test_download_fails_on_checksum_mismatch(tmp_path, small_df, monkeypatch):
    csv_bytes = small_df.to_csv(index=False).encode()
    monkeypatch.setattr(dl, "_fetch", lambda url, timeout: csv_bytes)
    sources = [
        {"name": "mirror", "url": "https://mirror/x.csv", "kind": "kaggle_csv", "sha256": "0" * 64}
    ]
    with pytest.raises(RuntimeError, match="sha256"):
        dl.download(_cfg_with(sources, None, tmp_path / "raw.csv", len(small_df)))


def test_zip_source_extracts_xls_member(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("default of credit card clients.xls", b"fake-xls")
    monkeypatch.setattr(dl, "_fetch", lambda url, timeout: buf.getvalue())
    monkeypatch.setattr(dl, "_read_xls", lambda data: pd.DataFrame({"payload": [data.decode()]}))
    out = dl.load_source({"name": "uci", "url": "u", "kind": "uci_zip", "inner_sha256": None}, 5)
    assert out["payload"][0] == "fake-xls"


def test_lfs_pointer_is_rejected(monkeypatch):
    class Resp:
        content = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(RuntimeError, match="LFS pointer"):
        dl._fetch("https://raw.githubusercontent.com/x", 5)
