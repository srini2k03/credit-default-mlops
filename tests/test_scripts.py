import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _metrics(source="uci"):
    test = {
        "roc_auc": 0.7,
        "pr_auc": 0.5,
        "f1": 0.5,
        "brier": 0.1,
        "business_cost": 0.6,
        "threshold": 0.2,
        "recall": 0.6,
        "approval_rate": 0.7,
    }
    return {
        "data": {
            "source": source,
            "synthetic": source != "uci",
            "n_rows": 100,
            "default_rate": 0.22,
            "n_test": 20,
        },
        "generated_at": "2026-01-01T00:00:00Z",
        "git_commit": "abcdef1234",
        "champion": "lightgbm",
        "cost": {"fn_cost": 5.0, "fp_cost": 1.0},
        "families": [{"family": "lightgbm", "test": test}, {"family": "logreg", "test": test}],
        "thresholds": [{"rule": "cost-optimal", **test}],
        "ablation": [{"family": "lightgbm", "strategy": "smote", "cv_pr_auc": 0.5}],
    }


def test_readme_block_is_replaced(tmp_path):
    mod = _load("update_readme_results")
    readme = tmp_path / "README.md"
    readme.write_text(f"before\n{mod.START}\nplaceholder\n{mod.END}\nafter\n")
    mod.update(readme, mod.render(_metrics()))
    text = readme.read_text()
    assert (
        "placeholder" not in text and "**lightgbm** (champion)" in text and text.endswith("after\n")
    )


def test_readme_refuses_synthetic_metrics():
    mod = _load("update_readme_results")
    with pytest.raises(SystemExit):
        mod.render(_metrics(source="synthetic"))


def test_locust_summary(tmp_path, capsys, monkeypatch):
    csv = tmp_path / "stats.csv"
    header = "Type,Name,Request Count,Failure Count,Requests/s,50%,95%,99%,Max Response Time"
    csv.write_text(
        f"{header}\nPOST,/predict,100,0,10.5,20,40,60,80\n,Aggregated,100,0,10.5,20,40,60,80\n"
    )
    spec = importlib.util.spec_from_file_location("summarize", ROOT / "locust" / "summarize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr("sys.argv", ["summarize.py", str(csv)])
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "| `POST /predict` | 100 | 0 | 10.5 | 20 | 40 | 60 | 80 |" in out and "**All**" in out
