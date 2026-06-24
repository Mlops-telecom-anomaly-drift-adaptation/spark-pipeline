"""
extra_metrics.py — Demo icin ek metrikler ve grafikler.

test_scores.npy cache'ini kullanir; SERVISE GITMEZ, saniyeler surer.
Once 'python adapt_v2.py' calismis ve test_scores.npy uretilmis olmali.

Uretir:
  Rakamlar (terminal):
    - PR-AUC (Average Precision)  <- dengesiz veride en dogru metrik
    - ROC-AUC
    - Anomali sinifinin P/R/F1     <- macro degil, asil zor sinif
    - Yanlis Alarm Orani (FPR) oncesi/sonrasi
    - PSI (drift buyuklugu)
  Grafikler (.png):
    - skor_dagilimi_normal_vs_anomali.png
    - skor_dagilimi_drift.png
    - esik_tarama_f1.png
    - pr_egrisi.png

Calistirma:
  $env:MLFLOW_TRACKING_URI = "file:./mlruns"
  python extra_metrics.py
"""
import os, numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve)

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


def psi(expected, actual, bins=10):
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None); a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def fpr(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return fp / (fp + tn) if (fp + tn) else float("nan")


print("Pencereler uretiliyor, cache okunuyor...")
Wte, meta = preprocess_and_window("test_rows.csv")
cur = np.load(CACHE)
if len(cur) != len(meta):
    raise SystemExit("Cache uyumsuz. 'python adapt_v2.py' ile yeniden uret.")
y = attach_labels(meta, "test_windows.csv")

# zaman-bolmeli (adapt_v2 ile ayni): kalibrasyon + degerlendirme
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

# ---------- RAKAMLAR ----------
print("\n" + "=" * 55)
print("EK METRIKLER")
print("=" * 55)
print(f"PR-AUC (Average Precision) : {average_precision_score(y, cur):.4f}   <- dengesiz veride asil metrik")
print(f"ROC-AUC                    : {roc_auc_score(y, cur):.4f}")

pred_old = (cur >= old_thr).astype(int)
print("\n-- Anomali sinifi (pozitif, etiket=1) | sabit fabrika esigi --")
print(f"  Precision: {precision_score(y, pred_old, pos_label=1, zero_division=0):.4f}"
      f"  Recall: {recall_score(y, pred_old, pos_label=1, zero_division=0):.4f}"
      f"  F1: {f1_score(y, pred_old, pos_label=1, zero_division=0):.4f}")

print("\n-- Yanlis Alarm Orani (FPR) | degerlendirme dilimi --")
print(f"  ONCESI (sabit esik {old_thr:.5f})   : {fpr(ev_y, (ev_s >= old_thr).astype(int)):.4f}")
print(f"  SONRASI (adaptif esik {best_thr:.5f}): {fpr(ev_y, (ev_s >= best_thr).astype(int)):.4f}")

print("\n-- PSI (drift buyuklugu) --")
ps = psi(cal_s, ev_s)
yorum = "anlamli drift" if ps > 0.2 else ("orta drift" if ps > 0.1 else "hafif")
print(f"  PSI (akisin erken vs gec yarisi): {ps:.4f}  ({yorum})")

# ---------- GRAFIKLER ----------
# 1) Normal vs anomali skor dagilimi
plt.figure(figsize=(8, 4))
plt.hist(cur[y == 0], bins=60, alpha=0.6, label="Normal", density=True)
plt.hist(cur[y == 1], bins=60, alpha=0.6, label="Anomali", density=True)
plt.axvline(old_thr, color="k", ls="--", lw=1.5, label=f"Fabrika esigi {old_thr:.4f}")
plt.axvline(best_thr, color="r", ls="-.", lw=1.5, label=f"Adaptif esik {best_thr:.4f}")
plt.title("Skor dagilimi: Normal vs Anomali"); plt.xlabel("anomali skoru"); plt.ylabel("yogunluk")
plt.legend(); plt.tight_layout(); plt.savefig("skor_dagilimi_normal_vs_anomali.png", dpi=130); plt.close()

# 2) Drift: erken vs gec akis
plt.figure(figsize=(8, 4))
plt.hist(cal_s, bins=60, alpha=0.6, label="Erken akis (kalibrasyon)", density=True)
plt.hist(ev_s, bins=60, alpha=0.6, label="Gec akis (degerlendirme)", density=True)
plt.title(f"Drift gorunumu — PSI={ps:.3f}"); plt.xlabel("anomali skoru"); plt.ylabel("yogunluk")
plt.legend(); plt.tight_layout(); plt.savefig("skor_dagilimi_drift.png", dpi=130); plt.close()

# 3) Esik tarama (degerlendirme diliminde macro-F1)
ths = np.linspace(ev_s.min(), np.quantile(ev_s, 0.999), 300)
f1s = [f1_score(ev_y, (ev_s >= t).astype(int), average="macro", zero_division=0) for t in ths]
plt.figure(figsize=(8, 4))
plt.plot(ths, f1s, lw=2)
plt.axvline(old_thr, color="k", ls="--", lw=1.5, label=f"Fabrika esigi ({old_thr:.4f})")
plt.axvline(best_thr, color="r", ls="-.", lw=1.5, label=f"Adaptif esik ({best_thr:.4f})")
plt.title("Esik tarama — macro-F1 (degerlendirme dilimi)"); plt.xlabel("esik"); plt.ylabel("macro-F1")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("esik_tarama_f1.png", dpi=130); plt.close()

# 4) PR egrisi
prec, rec, _ = precision_recall_curve(y, cur)
plt.figure(figsize=(6, 5))
plt.plot(rec, prec, lw=2, label=f"PR-AUC = {average_precision_score(y, cur):.3f}")
plt.title("Precision-Recall egrisi (tum test)"); plt.xlabel("Recall"); plt.ylabel("Precision")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("pr_egrisi.png", dpi=130); plt.close()

print("\nGrafikler kaydedildi: skor_dagilimi_normal_vs_anomali.png, "
      "skor_dagilimi_drift.png, esik_tarama_f1.png, pr_egrisi.png")
print("Bitti.")
