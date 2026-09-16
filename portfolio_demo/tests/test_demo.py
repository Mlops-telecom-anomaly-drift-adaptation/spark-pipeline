import json
import joblib
import pandas as pd
import pytest
from mlflow import MlflowClient
from portfolio_demo.run import Config, FEATURES, feature_shift, run_demo, sample


def test_synthetic_data_is_repeatable_and_shift_is_detectable():
    reference = sample(42, 1000)
    pd.testing.assert_frame_equal(reference, sample(42, 1000))
    assert max(feature_shift(reference, reference).values()) == 0
    assert max(feature_shift(reference, sample(43,1000,shifted=True)).values()) > .75


def test_end_to_end_logs_artifacts_and_uses_disjoint_splits(tmp_path):
    report = run_demo(tmp_path)
    manifest = json.loads((tmp_path/'artifacts/data_manifest.json').read_text())
    seen = set()
    for split in manifest.values():
        ids = set(split['row_ids'])
        assert not seen.intersection(ids)
        assert len(ids) == split['rows']
        seen.update(ids)
    assert report['drift_triggered']
    val = report['metrics']
    assert report['candidate_promoted'] == (val['candidate_validation']['f1'] >= val['baseline_validation']['f1']+.01)
    model = joblib.load(tmp_path/'artifacts/selected_model.joblib')
    assert len(model.predict(sample(99,10)[FEATURES])) == 10
    client = MlflowClient(tracking_uri=report['tracking_uri'])
    run = client.get_run(report['run_id'])
    assert run.info.status == 'FINISHED'
    assert run.data.metrics['selected_shifted_holdout_f1'] == report['metrics']['selected_shifted_holdout']['f1']
    assert {'results.json','data_manifest.json','selected_model.joblib'} <= {a.path for a in client.list_artifacts(report['run_id'])}


def test_no_drift_keeps_baseline(tmp_path):
    report = run_demo(tmp_path, Config(rows=600, drift_threshold=100))
    assert not report['drift_triggered']
    assert not report['candidate_promoted']
    assert 'candidate_validation' not in report['metrics']
    assert report['metrics']['baseline_shifted_holdout'] == report['metrics']['selected_shifted_holdout']


def test_invalid_size_fails_before_training(tmp_path):
    with pytest.raises(ValueError): run_demo(tmp_path, Config(rows=100))
