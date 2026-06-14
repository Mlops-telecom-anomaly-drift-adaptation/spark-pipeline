import requests, json, numpy as np, pandas as pd, torch

# config'ten feature kolonlarini ve pencere boyutunu al
cfg = torch.load("anomaly_model_config.pth", map_location="cpu", weights_only=False)
fcols, ws = cfg["feature_cols"], cfg["window_size"]

# inference_service.py'deki ayni on isleme + pencereleme
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
    wins, meta = [], []
    for cell, g in df.groupby("CELL", sort=False):
        g = g.sort_values("DATETIME").reset_index(drop=True)
        if len(g) < ws: continue
        X = g[fcols].to_numpy(np.float32)
        t = pd.to_datetime(g["DATETIME"]).dt.floor("s").to_numpy()
        for s in range(0, len(g)-ws+1):
            wins.append(X[s:s+ws]); meta.append({"CELL":str(cell), "END_TIME":pd.Timestamp(t[s+ws-1])})
    return np.asarray(wins, np.float32), pd.DataFrame(meta)

W, meta = preprocess_and_window("test_rows.csv", fcols, ws)
print("Pencere sayisi:", len(W))

# ilk 200 pencereyi gonder (hizli test)
flat = W[:200].reshape(len(W[:200]), -1).tolist()
payload = {"dataframe_split": {"data": flat}}
r = requests.post("http://127.0.0.1:5001/invocations",
                  headers={"Content-Type":"application/json"}, data=json.dumps(payload))
print("HTTP", r.status_code)
res = r.json()
print(json.dumps(res, indent=2)[:500])