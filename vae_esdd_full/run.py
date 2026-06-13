"""
VAE++ESDD — Ana Çalıştırma Dosyası

Kullanım:
  python run.py --dataset sea
  python run.py --dataset all
  python run.py --dataset forest --anomaly_rate 0.001
"""

import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model      import VAEplusESDD
from src.evaluation import PrequentialEvaluator
from src.data_all   import DATASET_MAP, DATASET_PARAMS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Grafik ────────────────────────────────────────────────────────────
def plot_results(ev, drift_times, drift_alarms, dataset_name):
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    fig.suptitle(f'VAE++ESDD — {dataset_name}\n'
                 'Neurocomputing 676 (2026)', fontsize=12, fontweight='bold')

    metrics = [
        (ev.gmean_hist,  'G-mean',       '#2E86AB'),
        (ev.recall_hist, 'Recall',        '#E84855'),
        (ev.spec_hist,   'Specificity',   '#3BB273'),
        (ev.pauc_hist,   'PAUC',          '#F6AE2D'),
    ]
    t = np.arange(len(ev.gmean_hist))
    smooth = lambda d: np.convolve(d, np.ones(300)/300, mode='same')

    for ax, (data, lbl, col) in zip(axes, metrics):
        ax.plot(t, data, alpha=0.15, color=col, lw=0.5)
        ax.plot(t, smooth(data), color=col, lw=2.0, label=lbl)
        for i, dt in enumerate(drift_times):
            ax.axvline(dt, color='black', ls='--', lw=1.5,
                       label='Gerçek Drift' if i==0 else '')
        for i, da in enumerate(drift_alarms):
            ax.axvline(da, color='red', ls=':', lw=1.5,
                       label='Alarm' if i==0 else '')
        ax.set_ylabel(lbl, fontsize=11)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8, ncol=3)

    axes[-1].set_xlabel('Zaman Adımı', fontsize=11)
    fig.legend(handles=[
        mpatches.Patch(color='black', alpha=0.8, label=f'Gerçek Drift {drift_times}'),
        mpatches.Patch(color='red',   alpha=0.8, label=f'Alarm {drift_alarms}'),
    ], loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    name = dataset_name.replace(' ','_').replace('/','_')
    path = os.path.join(RESULTS_DIR, f'{name}_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grafik → {path}")


# ── Tek dataset çalıştır ─────────────────────────────────────────────
def run_dataset(dataset_name, anomaly_rate, args):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name.upper()}")
    print(f"{'='*60}")

    # Veri yükle
    loader = DATASET_MAP[dataset_name]
    try:
        X_init, X_stream, y_stream, drift_times = loader(
            anomaly_rate=anomaly_rate, seed=args.seed)
    except FileNotFoundError as e:
        print(f"  ⚠️  {e}"); return None

    # Makale parametrelerini al
    dp = DATASET_PARAMS[dataset_name]

    # Model oluştur
    model = VAEplusESDD(
        input_dim   = X_stream.shape[1],
        hidden_dims = tuple(args.hidden_dims or dp['hidden_dims']),
        latent_dim  = args.latent_dim or dp['latent_dim'],
        n           = args.n_ensemble,
        W_train     = args.w_train,
        gamma       = args.gamma,
        W_drift_min = args.w_drift_min,
        W_drift_max = args.w_drift_max,
        P_thre      = args.p_thre,
        D_thre      = args.d_thre,
        P_warn      = args.p_warn,
        P_alarm     = args.p_alarm,
        expiry_time = args.expiry_time,
        beta        = args.beta,
        lr          = args.lr or dp['lr'],
        num_epochs  = args.num_epochs or dp['epochs'],
        batch_size  = args.batch_size,
        seed        = args.seed,
    )

    model.initialize(X_init)

    # Stream
    ev = PrequentialEvaluator(fading=0.99, pauc_window=1000)
    preds, scores = [], []
    log_every = max(500, len(X_stream)//20)

    print(f"\n  Stream: {len(X_stream):,} örnek...")
    for step, (xi, yi) in enumerate(zip(X_stream, y_stream)):
        pred, score, _ = model.process(xi)
        ev.update(int(yi), pred, score)
        preds.append(pred); scores.append(score)
        if (step+1) % log_every == 0:
            gm = np.mean(ev.gmean_hist[-log_every:])
            rc = np.mean(ev.recall_hist[-log_every:])
            print(f"    [t={step+1:>7,}] G-mean={gm:.3f} Recall={rc:.3f} "
                  f"Alarm={len(model.drift_alarms)}")

    metrics = ev.print_summary(drift_times, model.drift_alarms)
    plot_results(ev, drift_times, model.drift_alarms, dataset_name)

    # Kaydet
    pd.DataFrame({
        'y_true': y_stream, 'y_pred': preds, 'score': scores,
        'gmean': ev.gmean_hist, 'recall': ev.recall_hist,
        'specificity': ev.spec_hist, 'pauc': ev.pauc_hist,
    }).to_csv(os.path.join(RESULTS_DIR, f'{dataset_name}_stream.csv'), index=False)

    return {
        'dataset'       : dataset_name,
        'anomaly_rate'  : anomaly_rate,
        **metrics,
        'drift_alarms'  : str(model.drift_alarms),
        'true_drifts'   : str(drift_times),
    }


# ── Ana ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='VAE++ESDD')

    p.add_argument('--dataset', default='sea',
                   choices=list(DATASET_MAP.keys()) + ['all'],
                   help='Dataset adı veya "all"')
    p.add_argument('--anomaly_rate', type=float, default=0.01,
                   help='Anomali oranı (0.01=%1 severe, 0.001=%0.1 extreme)')

    # Model — varsayılanlar makale Tablo 3'ten
    p.add_argument('--hidden_dims',  type=int, nargs='+', default=None,
                   help='Gizli katmanlar (default: dataset özelinde)')
    p.add_argument('--latent_dim',   type=int,   default=None)
    p.add_argument('--n_ensemble',   type=int,   default=10)
    p.add_argument('--w_train',      type=int,   default=3000)
    p.add_argument('--gamma',        type=int,   default=2000)
    p.add_argument('--w_drift_min',  type=int,   default=180)
    p.add_argument('--w_drift_max',  type=int,   default=220)
    p.add_argument('--p_thre',       type=int,   default=1)
    p.add_argument('--d_thre',       type=int,   default=10)
    p.add_argument('--p_warn',       type=float, default=0.01)
    p.add_argument('--p_alarm',      type=float, default=0.001)
    p.add_argument('--expiry_time',  type=int,   default=100)
    p.add_argument('--beta',         type=float, default=1.0)
    p.add_argument('--lr',           type=float, default=None,
                   help='Öğrenme hızı (default: dataset özelinde)')
    p.add_argument('--num_epochs',   type=int,   default=None,
                   help='Epoch sayısı (default: dataset özelinde)')
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--seed',         type=int,   default=42)

    args = p.parse_args()

    datasets = list(DATASET_MAP.keys()) if args.dataset == 'all' else [args.dataset]

    all_results = []
    for ds in datasets:
        result = run_dataset(ds, args.anomaly_rate, args)
        if result:
            all_results.append(result)

    # Özet tablo
    if all_results:
        df = pd.DataFrame(all_results)
        path = os.path.join(RESULTS_DIR, 'summary_all.csv')
        df.to_csv(path, index=False)
        print(f"\n{'='*60}")
        print("  ÖZET TABLO")
        print(f"{'='*60}")
        print(df[['dataset','G-mean','Recall','Specificity','PAUC']].to_string(index=False))
        print(f"\n  Özet → {path}")


if __name__ == '__main__':
    main()
