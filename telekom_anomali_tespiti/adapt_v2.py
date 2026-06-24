"""
adapt_v2.py — Sizintisiz (leak-free) zaman-bolmeli adaptif esik.

Neden bu yaklasim:
  train_windows.csv'de etiketli anomali YOK (model sadece normal veriyle egitildi).
  Bu yuzden F1-optimal esigi referans veride kalibre edemiyoruz.
  Cozum: test akisini ZAMANA gore ikiye boluyoruz.
    - Erken dilim  -> canlida biriken etiketli geri-bildirim (kalibrasyon)
    - Gec dilim    -> kalibrasyonda hic gorulmedi (sizintisiz degerlendirme)
  Adaptif esik erken dilimde F1'e gore secilir, SADECE gec dilimde olculur.
  Bu, gercek bir test-time adaptation senaryosudur ve veri sizintisi icermez.

Calistirma (servis 5001'de ayakta olmali):
  $env:MLFLOW_TRACKING_URI = "file:./mlruns"
  python adapt_v2.py
Ilk kosu skorlari servisten ceker (birkac dk) ve test_scores.npy'ye cache'ler;
sonraki kosular aninda calisir.
"""
import os, json, requests, numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

URL = "http://127.0.0.1:5001/invocations"
CACHE = "test_scores.npy"

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


def get_scores(W):
    if os.path.exists(CACHE):
        arr = np.load(CACHE)
        if len(arr) == len(W):
            print(f"  (skorlar cache'ten yuklendi: {CACHE})")
            return arr
    out = []
    for i in range(0, len(W), 1000):
        flat = W[i:i + 1000].reshape(len(W[i:i + 1000]), -1).tolist()
        r = requests.post(URL, headers={"Content-Type": "application/json"},
                          data=json.dumps({"dataframe_split": {"data": flat}}))
        out += [p["anomaly_score"] for p in r.json()["predictions"]]
    arr = np.array(out)
    np.save(CACHE, arr)
    return arr


def attach_labels(meta, labels_csv):
    xl = pd.read_csv(labels_csv)
    xl["CELL"] = xl["CELL"].astype(str).str.strip()
    xl["END_TIME"] = pd.to_datetime(xl["END_TIME"], errors="coerce").dt.floor("s")
    xl = xl.dropna(subset=["CELL", "END_TIME", "label"])
    xl["_L"] = xl["label"].astype(int)
    lut = xl.groupby(["CELL", "END_TIME"])["_L"].max().to_dict()
    me = pd.to_datetime(meta["END_TIME"]).dt.floor("s")
    return np.array([lut.get((c, t), 0) for c, t in zip(meta["CELL"], me)], dtype=int)


def report(name, thr, s, yy):
    pred = (s >= thr).astype(int)
    print(f"\n--- {name} (esik={thr:.6f}) ---")
    print("Precision:", round(precision_score(yy, pred, average="macro", zero_division=0), 4),
          "Recall:", round(recall_score(yy, pred, average="macro", zero_division=0), 4),
          "F1:", round(f1_score(yy, pred, average="macro", zero_division=0), 4))
    print("Confusion:\n", confusion_matrix(yy, pred))


print("Test pencereleri uretiliyor...")
Wte, meta = preprocess_and_window("test_rows.csv")
print("Skorlar aliniyor...")
cur = get_scores(Wte)
y = attach_labels(meta, "test_windows.csv")

# --- Akisi zamana gore sirala ve ikiye bol ---
order = np.argsort(pd.to_datetime(meta["END_TIME"]).to_numpy())
cur_o, y_o = cur[order], y[order]
split = len(cur_o) // 2
cal_s, cal_y = cur_o[:split], y_o[:split]   # kalibrasyon (etiketli geri-bildirim)
ev_s, ev_y = cur_o[split:], y_o[split:]     # degerlendirme (sizintisiz)

print(f"\nKalibrasyon penceresi: {len(cal_s)} (anomali: {int(cal_y.sum())})")
print(f"Degerlendirme penceresi: {len(ev_s)} (anomali: {int(ev_y.sum())})")

# --- Adaptif esik: SADECE kalibrasyon diliminde macro-F1'i maksimize et ---
cand = np.unique(np.quantile(cal_s, np.linspace(0.50, 0.999, 400)))
best_thr, best_f1 = old_thr, -1.0
for t in cand:
    f = f1_score(cal_y, (cal_s >= t).astype(int), average="macro", zero_division=0)
    if f > best_f1:
        best_f1, best_thr = f, float(t)
new_thr = best_thr

print("\n===== ADAPTASYON (zaman-bolmeli, sizintisiz) =====")
print(f"Kalibrasyonda secilen adaptif esik: {new_thr:.6f} (kalibrasyon F1={best_f1:.4f})")
print(">> Asagidaki iki sonuc SADECE degerlendirme diliminde (kalibrasyonda gorulmeyen veri):")
report("ONCESI (sabit fabrika esigi)", old_thr, ev_s, ev_y)
report("SONRASI (drift sonrasi adaptif esik)", new_thr, ev_s, ev_y)
