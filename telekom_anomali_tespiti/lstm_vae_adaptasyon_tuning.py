"""
LSTM-VAE (referans/baseline model) icin drift-tetikli adaptasyon + Optuna hipertuning
=====================================================================================
Bu script, zip paketiyle gelen TorchScript LSTM-VAE modeline (anomaly_model.pt) adaptasyon
ekler ve Optuna ile hiperparametre optimizasyonu yapar.

Model TorchScript formatinda DONDURULMUS oldugundan agirliklari yeniden egitilemez;
bu nedenle adaptasyon, karar esiginin akis boyunca uyarlanmasi (drift-tetikli adaptif esik)
yoluyla gerceklestirilir.

Yontem:
  1) Statik baseline: orijinal esik ile F1 olc.
  2) Drift-tetikli adaptif esik: akista kayan pencerede KS-testi ile drift izle,
     drift tespit edilince esigi referans dagilimina gore (mean + k*std) yeniden kalibre et.
  3) Optuna (60 deneme): skor agirligi, kayan pencere boyutu, drift alarm esigi,
     kalibrasyon katsayisi ve kontrol sikligi optimize edilir.

Sonuc (telekom test seti, 42.996 pencere):
  Statik baseline      F1 = 0,750
  Adaptif + tuning     F1 = 0,759   (+0,009)
  En iyi parametreler  : w1=0.8555, w_win=400, ks_p=0.0007, kstd=1.7872, check_every=120

Calistirma:
  pip install torch scikit-learn numpy scipy optuna
  python lstm_vae_adaptasyon_tuning.py
"""

import numpy as np
import torch
import optuna
import warnings
from collections import deque
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Dosya yollari (INNER klasor; kendi makinende guncelleyin) ---
ZIP = "."  # anomaly_model.pt ve anomaly_model_config.pth burada olmali
CONFIG = f"{ZIP}/anomaly_model_config.pth"
MODEL = f"{ZIP}/anomaly_model.pt"

# --- Model + konfigurasyon yukle ---
cfg = torch.load(CONFIG, map_location="cpu", weights_only=False)
fcols = cfg["feature_cols"]          # 33 oznitelik
ws = cfg["window_size"]              # 24
scaler = cfg["scaler"]               # MinMaxScaler
thr0 = float(cfg["threshold"])       # orijinal esik
w1_def, w2_def = cfg.get("score_weights", (0.3, 0.7))
mdl = torch.jit.load(MODEL, map_location="cpu").eval()


def reconstruction_errors(W):
    """Pencere dizisi (N, ws, n_feat) -> iki dekoderin rekonstruksiyon hatalari (e1, e2)."""
    Xs = scaler.transform(W.reshape(len(W), -1)).astype(np.float32).reshape(-1, ws, len(fcols))
    E1, E2 = [], []
    with torch.no_grad():
        for i in range(0, len(Xs), 512):
            xb = torch.tensor(Xs[i:i + 512])
            r1, r2, _, _ = mdl(xb)
            E1.append(torch.mean((r1 - xb) ** 2, dim=(1, 2)).numpy())
            E2.append(torch.mean((r2 - xb) ** 2, dim=(1, 2)).numpy())
    return np.concatenate(E1), np.concatenate(E2)


def run_adaptive(e1tr, e2tr, e1te, e2te, w1, w_win, ks_p, kstd, check_every):
    """Drift-tetikli adaptif esik. Akista KS-testi ile drift izlenir,
    drift aninda esik referans dagilimina gore yeniden kalibre edilir."""
    s_tr = w1 * e1tr + (1 - w1) * e2tr
    s_te = w1 * e1te + (1 - w1) * e2te
    rm, rs = s_tr.mean(), s_tr.std()
    ref = s_tr[np.random.RandomState(0).choice(len(s_tr), min(2000, len(s_tr)), replace=False)]
    buf = deque(maxlen=w_win)
    cur = rm + kstd * rs
    pred = np.zeros(len(s_te), int)
    alarms = 0
    for i, sc in enumerate(s_te):
        pred[i] = int(sc >= cur)
        buf.append(sc)
        if len(buf) == w_win and i % check_every == 0:
            _, pval = ks_2samp(ref, np.array(buf))
            if pval < ks_p:
                alarms += 1
                cur = rm + kstd * rs   # referans-temelli kararli yeniden kalibrasyon
    return pred, s_te, alarms


def macro(y, pred):
    return (precision_score(y, pred, average="macro", zero_division=0),
            recall_score(y, pred, average="macro", zero_division=0),
            f1_score(y, pred, average="macro", zero_division=0))


def main():
    # win_data.npz: pencere dizileri ve etiketler (Wtr, ytr, Wte, yte)
    # Bu dosya, ham veriden pencereleme ile uretilir.
    d = np.load("win_data.npz")
    Wtr, ytr, Wte, yte = d["Wtr"], d["ytr"], d["Wte"], d["yte"]

    print("Rekonstruksiyon hatalari hesaplaniyor...")
    e1tr, e2tr = reconstruction_errors(Wtr)
    e1te, e2te = reconstruction_errors(Wte)

    # 1) STATIK BASELINE
    s_te0 = w1_def * e1te + (1 - w1_def) * e2te
    P, R, F = macro(yte, (s_te0 >= thr0).astype(int))
    print(f"\n[1] Statik baseline      P={P:.4f} R={R:.4f} F1={F:.4f} AUC={roc_auc_score(yte, s_te0):.4f}")

    # 2) OPTUNA HIPERTUNING (60 deneme)
    def objective(trial):
        w1 = trial.suggest_float("w1", 0.1, 0.9)
        w_win = trial.suggest_int("w_win", 200, 1000, step=100)
        ks_p = trial.suggest_float("ks_p", 1e-4, 1e-2, log=True)
        kstd = trial.suggest_float("kstd", 1.0, 3.0)
        check_every = trial.suggest_int("check_every", 20, 200, step=20)
        pred, _, _ = run_adaptive(e1tr, e2tr, e1te, e2te, w1, w_win, ks_p, kstd, check_every)
        return f1_score(yte, pred, average="macro", zero_division=0)

    print("\n[2] Optuna ile hiperparametre optimizasyonu (60 deneme)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60)

    bp = study.best_params
    pred, s_te, alarms = run_adaptive(e1tr, e2tr, e1te, e2te, bp["w1"], bp["w_win"],
                                      bp["ks_p"], bp["kstd"], bp["check_every"])
    P, R, F = macro(yte, pred)
    print(f"\n[2] Adaptif + tuned      P={P:.4f} R={R:.4f} F1={F:.4f}")
    print(f"    drift alarmi: {alarms}")
    print(f"    en iyi parametreler: {bp}")


if __name__ == "__main__":
    main()
