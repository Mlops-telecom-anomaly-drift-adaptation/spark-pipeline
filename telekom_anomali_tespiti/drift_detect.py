import requests, json, numpy as np, pandas as pd, torch
from scipy import stats

cfg = torch.load("anomaly_model_config.pth", map_location="cpu", weights_only=False)
fcols, ws = cfg["feature_cols"], cfg["window_size"]

def preprocess_and_window(csv_path, fcols, ws):
    df = pd.read_csv(csv_path)
    df = df.dropna(axis="columns", how="all")
    df = df.drop(columns=[c for c in df.columns if "PRB" in str(c).upper()], errors="ignore")
    df["CELL"] = df["CELL"].astype(str).str.strip()
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    df = df.dropna(subset=["DATETIME"]).sort_values(["CELL","DATETIME"]).reset_index(drop=True)
    for c in fcols:
        if c not in df.columns: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[fcols] = df[fcols].fillna(df[fcols].median())
    wins = []
    for cell, g in df.groupby("CELL", sort=False):
        g = g.sort_values("DATETIME").reset_index(drop=True)
        if len(g) < ws: continue
        X = g[fcols].to_numpy(np.float32)
        for s in range(0, len(g)-ws+1):
            wins.append(X[s:s+ws])
    return np.asarray(wins, np.float32)

def get_scores(W):
    out = []
    for i in range(0, len(W), 1000):
        flat = W[i:i+1000].reshape(len(W[i:i+1000]), -1).tolist()
        r = requests.post("http://127.0.0.1:5001/invocations",
                          headers={"Content-Type":"application/json"},
                          data=json.dumps({"dataframe_split":{"data":flat}}))
        out += [p["anomaly_score"] for p in r.json()["predictions"]]
    return np.array(out)

print("Referans (train) pencereleri uretiliyor...")
Wtr = preprocess_and_window("train_rows.csv", fcols, ws)
print("Test pencereleri uretiliyor...")
Wte = preprocess_and_window("test_rows.csv", fcols, ws)

print("Skorlar aliniyor (referans)...")
ref = get_scores(Wtr)
print("Skorlar aliniyor (test)...")
cur = get_scores(Wte)

# Drift testi: KS testi + ortalama kaymasi
ks_stat, p_val = stats.ks_2samp(ref, cur)
print("\n===== DATA DRIFT RAPORU =====")
print(f"Referans skor ort: {ref.mean():.5f} | std: {ref.std():.5f}")
print(f"Test     skor ort: {cur.mean():.5f} | std: {cur.std():.5f}")
print(f"KS istatistigi: {ks_stat:.4f} | p-deger: {p_val:.3e}")
print("DRIFT VAR" if p_val < 0.05 else "DRIFT YOK", f"(p {'<' if p_val<0.05 else '>='} 0.05)")