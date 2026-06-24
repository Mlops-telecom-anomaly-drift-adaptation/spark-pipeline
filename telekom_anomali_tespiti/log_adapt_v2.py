"""
log_adapt_v2.py — Sizintisiz zaman-bolmeli adaptasyon sonucunu MLflow'a loglar.

adapt_v2.py'nin urettigi test_scores.npy cache'ini kullanir; SERVISE GITMEZ,
saniyeler surer. Degerlendirme (gec) diliminde iki run loglar:
  - LSTM-VAE-Sabit-Esik-EvalDilim         (fabrika esigi)
  - LSTM-VAE-Adaptif-Sizintisiz-EvalDilim (drift sonrasi adaptif esik)
Boylece MLflow Compare ekraninda adaptasyon kazanci da gorunur.

Calistirma (once adapt_v2.py calismis ve test_scores.npy uretilmis olmali):
  $env:MLFLOW_TRACKING_URI = "file:./mlruns"
  python log_adapt_v2.py
"""
import os, numpy as np, pandas as pd, torch, mlflow
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)

CACHE = "test_scores.npy"
if not os.path.exists(CACHE):
    raise SystemExit("test_scores.npy yok. Once 'python adapt_v2.py' calistir.")

cfg = torch.load("anomaly_model_config.pth", map_location="cpu", weights_only=False)
fcols, ws = cfg["feature_cols"], cfg["window_size"]
old_thr = float(cfg["threshold"])


def preprocess_and_window(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(axis="columns", how="all")
    df = df.drop(columns=[c for c in df.columns if "PRB" in str(c).upper()], errors="ignore")
    df["CELL"] = df["CELL"].astype(str).str.strip()
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    df = df.dropna(subset=["DATETIME"]).sort_values(["CELL", "DATETIME"]).reset_index(drop=True)
    for c in fcols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[fcols] = df[fcols].fillna(df[fcols].median())
    wins, meta = [], []
    for cell, g in df.groupby("CELL", sort=False):
        g = g.sort_values("DATETIME").reset_index(drop=True)
        if len(g) < ws:
            continue
        X = g[fcols].to_numpy(np.float32)
        t = pd.to_datetime(g["DATETIME"]).dt.floor("s").to_numpy()
        for s in range(0, len(g) - ws + 1):
            wins.append(X[s:s + ws])
            meta.append({"CELL": str(cell), "END_TIME": pd.Timestamp(t[s + ws - 1])})
    return np.asarray(wins, np.float32), pd.DataFrame(meta)


def attach_labels(meta, labels_csv):
    xl = pd.read_csv(labels_csv)
    xl["CELL"] = xl["CELL"].astype(str).str.strip()
    xl["END_TIME"] = pd.to_datetime(xl["END_TIME"], errors="coerce").dt.floor("s")
    xl = xl.dropna(subset=["CELL", "END_TIME", "label"])
    xl["_L"] = xl["label"].astype(int)
    lut = xl.groupby(["CELL", "END_TIME"])["_L"].max().to_dict()
    me = pd.to_datetime(meta["END_TIME"]).dt.floor("s")
    return np.array([lut.get((c, t), 0) for c, t in zip(meta["CELL"], me)], dtype=int)


print("Pencereler uretiliyor, cache okunuyor...")
Wte, meta = preprocess_and_window("test_rows.csv")
cur = np.load(CACHE)
if len(cur) != len(meta):
    raise SystemExit("Cache uyumsuz. 'python adapt_v2.py' ile yeniden uret.")
y = attach_labels(meta, "test_windows.csv")

order = np.argsort(pd.to_datetime(meta["END_TIME"]).to_numpy())
cur_o, y_o = cur[order], y[order]
split = len(cur_o) // 2
cal_s, cal_y = cur_o[:split], y_o[:split]
ev_s, ev_y = cur_o[split:], y_o[split:]

cand = np.unique(np.quantile(cal_s, np.linspace(0.50, 0.999, 400)))
best_thr, best_f1 = old_thr, -1.0
for t in cand:
    f = f1_score(cal_y, (cal_s >= t).astype(int), average="macro", zero_division=0)
    if f > best_f1:
        best_f1, best_thr = f, float(t)


def metrics(thr, s, yy):
    p = (s >= thr).astype(int)
    return {
        "macro_precision": precision_score(yy, p, average="macro", zero_division=0),
        "macro_recall": recall_score(yy, p, average="macro", zero_division=0),
        "macro_f1": f1_score(yy, p, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(yy, s) if len(np.unique(yy)) > 1 else float("nan"),
    }


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("telekom-model-karsilastirma")

runs = [
    ("LSTM-VAE-Sabit-Esik-EvalDilim", old_thr, "sabit_esik_eval"),
    ("LSTM-VAE-Adaptif-Sizintisiz-EvalDilim", best_thr, "adaptif_sizintisiz_eval"),
]
for name, thr, mod in runs:
    m = metrics(thr, ev_s, ev_y)
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", "LSTM-VAE")
        mlflow.log_param("mod", mod)
        mlflow.log_param("esik", round(thr, 6))
        mlflow.log_param("degerlendirme", "zaman-bolmeli sizintisiz (gec dilim)")
        mlflow.log_param("eval_pencere_sayisi", int(len(ev_s)))
        mlflow.log_param("eval_anomali", int(ev_y.sum()))
        for k, v in m.items():
            mlflow.log_metric(k, float(v))
        print("loglandi:", name, {k: round(v, 4) for k, v in m.items()})
print("Bitti. MLflow'da 'telekom-model-karsilastirma' deneyini yenile.")
