"""Train, detect feature shift, validate a candidate, then evaluate a locked holdout."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from mlflow import MlflowClient

FEATURES = ['latency_ms', 'packet_loss_pct', 'throughput_mbps']


@dataclass(frozen=True)
class Config:
    seed: int = 42
    rows: int = 2400
    drift_threshold: float = 0.75
    min_f1_gain: float = 0.01


def sample(seed, rows, shifted=False, prefix='base'):
    """Illustrative KPIs and synthetic labels; not a simulator of a real network."""
    rng = np.random.default_rng(seed)
    latency = np.clip(rng.normal(50 + 18*shifted, 12, rows), 1, None)
    loss = np.clip(rng.beta(2, 20, rows)*100 + 3*shifted, 0, 100)
    throughput = rng.lognormal(4, .4, rows) * (.65 if shifted else 1)
    noise = rng.normal(0, .7, rows)
    # The shifted window changes both feature distribution and the label rule.
    score = (.08 + .06*shifted)*(latency-50) + (.6-.25*shifted)*(loss-8) - .035*(throughput-50) + noise
    df = pd.DataFrame({'latency_ms': latency, 'packet_loss_pct': loss, 'throughput_mbps': throughput,
                       'anomaly': (score > 2.5).astype(int), 'row_id': [f'{prefix}-{i}' for i in range(rows)]})
    return df


def feature_shift(reference, current):
    scale = reference[FEATURES].std(ddof=0).replace(0, 1)
    return ((current[FEATURES].mean()-reference[FEATURES].mean()).abs()/scale).to_dict()


def train(data, seed):
    model = RandomForestClassifier(n_estimators=80, max_depth=7, min_samples_leaf=3,
                                   random_state=seed, n_jobs=1, class_weight='balanced')
    model.fit(data[FEATURES], data.anomaly)
    return model


def evaluate(model, data):
    pred = model.predict(data[FEATURES])
    probability = model.predict_proba(data[FEATURES])[:, 1]
    return {'f1': float(f1_score(data.anomaly, pred, zero_division=0)),
            'precision': float(precision_score(data.anomaly, pred, zero_division=0)),
            'recall': float(recall_score(data.anomaly, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(data.anomaly, probability))}


def fingerprint(df):
    return hashlib.sha256(df.to_csv(index=False, float_format='%.12g').encode()).hexdigest()


def run_demo(output_dir, config=Config()):
    if config.rows < 500:
        raise ValueError('Use at least 500 rows per window for the demonstration.')
    if config.drift_threshold <= 0 or config.min_f1_gain < 0:
        raise ValueError('Threshold must be positive; minimum gain must be nonnegative.')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = sample(config.seed, config.rows, prefix='reference')
    shifted = sample(config.seed+1, config.rows, shifted=True, prefix='shifted')
    split = int(config.rows*.7)
    cal_end, val_end = int(config.rows*.5), int(config.rows*.75)
    parts = {'train': base.iloc[:split], 'baseline_test': base.iloc[split:],
             'calibration': shifted.iloc[:cal_end], 'validation': shifted.iloc[cal_end:val_end],
             'holdout': shifted.iloc[val_end:]}
    baseline = train(parts['train'], config.seed)
    shift = feature_shift(parts['train'], parts['calibration'])
    triggered = max(shift.values()) > config.drift_threshold
    baseline_validation = evaluate(baseline, parts['validation'])
    candidate = train(parts['calibration'], config.seed) if triggered else None
    candidate_validation = evaluate(candidate, parts['validation']) if candidate else None
    promote = bool(candidate and candidate_validation['f1'] >= baseline_validation['f1']+config.min_f1_gain)
    selected = candidate if promote else baseline
    # Freeze selection before evaluating the holdout; do not tune on these scores.
    metrics = {'baseline_test': evaluate(baseline, parts['baseline_test']),
               'baseline_validation': baseline_validation,
               'baseline_shifted_holdout': evaluate(baseline, parts['holdout']),
               'selected_shifted_holdout': evaluate(selected, parts['holdout'])}
    if candidate:
        metrics['candidate_validation'] = candidate_validation
    manifest = {name: {'rows':len(df), 'sha256':fingerprint(df), 'row_ids':df.row_id.tolist()} for name,df in parts.items()}
    report = {'config':asdict(config), 'features':FEATURES, 'data':'synthetic demonstration only',
              'drift_scores':shift, 'drift_triggered':triggered, 'candidate_promoted':promote,
              'selection_rule':'validation F1 gain >= min_f1_gain; holdout excluded from selection',
              'metrics':metrics, 'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','mlflow-skinny']}}
    artifacts = output/'artifacts'; artifacts.mkdir(exist_ok=True)
    (artifacts/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (artifacts/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    joblib.dump(selected, artifacts/'selected_model.joblib')
    tracking = output/'mlruns'
    # Explicit local tracking/artifact URIs prevent environment variables selecting a remote server.
    client = MlflowClient(tracking_uri=tracking.as_uri(), registry_uri=tracking.as_uri())
    experiment = client.get_experiment_by_name('synthetic-telecom-demo')
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        'synthetic-telecom-demo', artifact_location=(output/'mlartifacts').as_uri())
    run = client.create_run(experiment_id, tags={'data_kind':'synthetic', 'purpose':'portfolio demo'})
    try:
        for key,value in asdict(config).items(): client.log_param(run.info.run_id,key,value)
        for group,scores in metrics.items():
            for key,value in scores.items(): client.log_metric(run.info.run_id,f'{group}_{key}',value)
        for key,value in shift.items(): client.log_metric(run.info.run_id,f'shift_{key}',value)
        client.log_metric(run.info.run_id,'candidate_promoted',int(promote))
        client.log_artifacts(run.info.run_id,str(artifacts))
        client.set_terminated(run.info.run_id)
    except Exception:
        client.set_terminated(run.info.run_id,status='FAILED')
        raise
    report['run_id'] = run.info.run_id
    report['tracking_uri'] = tracking.as_uri()
    (output/'run.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='portfolio_demo/output')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    print(json.dumps(run_demo(args.output,Config(seed=args.seed)),indent=2))
