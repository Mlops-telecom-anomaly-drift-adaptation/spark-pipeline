import requests, json, numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

cfg = torch.load("anomaly_model_config.pth", map_location="cpu", weights_only=False)
fcols, ws = cfg["feature_cols"], cfg["window_size"]
old_thr = float(cfg["threshold"])

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
            wins.append(X[s:s+ws]); meta.append({"CELL":str(cell),"END_TIME":pd.Timestamp(t[s+ws-1])})
    return np.asarray(wins, np.float32), pd.DataFrame(meta)

def get_scores(W):
    out=[]
    for i in range(0,len(W),1000):
        flat=W[i:i+1000].reshape(len(W[i:i+1000]),-1).tolist()
        r=requests.post("http://127.0.0.1:5001/invocations",headers={"Content-Type":"application/json"},
                        data=json.dumps({"dataframe_split":{"data":flat}}))
        out+=[p["anomaly_score"] for p in r.json()["predictions"]]
    return np.array(out)

def attach_labels(meta, labels_csv):
    xl=pd.read_csv(labels_csv); xl["CELL"]=xl["CELL"].astype(str).str.strip()
    xl["END_TIME"]=pd.to_datetime(xl["END_TIME"],errors="coerce").dt.floor("s")
    xl=xl.dropna(subset=["CELL","END_TIME","label"]); xl["_L"]=xl["label"].astype(int)
    lut=xl.groupby(["CELL","END_TIME"])["_L"].max().to_dict()
    me=pd.to_datetime(meta["END_TIME"]).dt.floor("s")
    return np.array([lut.get((c,t),0) for c,t in zip(meta["CELL"],me)],dtype=int)

# referans skorlari -> yeni esik (referansin %95 persentili)
Wtr,_ = preprocess_and_window("train_rows.csv", fcols, ws)
ref = get_scores(Wtr)
new_thr = float(np.quantile(ref, 0.95))

# test skorlari + etiketler
Wte, meta = preprocess_and_window("test_rows.csv", fcols, ws)
cur = get_scores(Wte)
y = attach_labels(meta, "test_windows.csv")

def report(name, thr):
    pred=(cur>=thr).astype(int)
    print(f"\n--- {name} (esik={thr:.6f}) ---")
    print("Precision:",round(precision_score(y,pred,average='macro',zero_division=0),4),
          "Recall:",round(recall_score(y,pred,average='macro',zero_division=0),4),
          "F1:",round(f1_score(y,pred,average='macro',zero_division=0),4))
    print("Confusion:\n",confusion_matrix(y,pred))

print("===== ADAPTASYON =====")
report("Adaptasyon ONCESI (orijinal esik)", old_thr)
report("Adaptasyon SONRASI (referans %95)", new_thr)