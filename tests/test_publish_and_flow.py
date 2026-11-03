import pytest

from credit_default.pipeline.publish import SyntheticPublishError, publish


def test_publish_refuses_synthetic_runs(trained, cfg, tmp_path):
    with pytest.raises(SyntheticPublishError):
        publish(
            cfg,
            run_dir=trained.out_dir,
            results_dir=tmp_path / "results",
            bundle_dir=tmp_path / "bundled",
        )
    assert not (tmp_path / "results").exists()


def test_flow_module_wires_all_steps():
    pytest.importorskip("prefect")
    import flows.training_flow as f

    assert f.training_flow.name == "credit-default-training"
    for t in (f.ingest, f.validate, f.train, f.evaluate, f.register):
        assert callable(t.fn)
