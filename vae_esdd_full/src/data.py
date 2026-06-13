# src/data.py
"""
Veri Yükleme Modülü

Desteklenen kaynaklar:
    1. Sea  — sentetik, recurrent drift (Makale Tablo 2)
    2. CSV  — kendi veri setin (son sütun = etiket)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════
#  SEA DATASET (Makale Tablo 2)
# ══════════════════════════════════════════════════════════════════════
def load_sea(n_total=20000, anomaly_rate=0.01,
             drift1=10000, drift2=15000, seed=42):
    """
    Sea sentetik veri seti.

    Kavram A (t=0..drift1-1, t=drift2..son):
        Normal  : x1 + x2 > 7
        Anomali : x1 + x2 ≤ 7

    Kavram B (t=drift1..drift2-1):
        Normal  : x1 + x2 ≤ 7   ← ANİ DRIFT
        Anomali : x1 + x2 > 7

    Döndürür:
        X_init (np.ndarray) : ilk eğitim verisi (2000 normal örnek)
        X_stream (np.ndarray): stream verisi
        y_stream (np.ndarray): etiketler (0=normal, 1=anomali)
        drift_times (list)   : [drift1, drift2]
    """
    rng = np.random.RandomState(seed)

    def _segment(n, concept, anomaly_rate, rng):
        n_anom   = max(1, int(n * anomaly_rate))
        n_normal = n - n_anom
        pts = rng.uniform(0, 10, (n * 30, 2))
        if concept == 'A':
            nm = pts[:, 0] + pts[:, 1] > 7
            am = pts[:, 0] + pts[:, 1] <= 7
        else:
            nm = pts[:, 0] + pts[:, 1] <= 7
            am = pts[:, 0] + pts[:, 1] > 7
        normals  = pts[nm][:n_normal]
        anomalies = pts[am][:n_anom]
        X = np.vstack([normals, anomalies])
        y = np.concatenate([np.zeros(len(normals)), np.ones(len(anomalies))])
        idx = rng.permutation(len(X))
        return X[idx], y[idx]

    Xa, ya = _segment(drift1,           'A', anomaly_rate, rng)
    Xb, yb = _segment(drift2 - drift1,  'B', anomaly_rate, rng)
    Xc, yc = _segment(n_total - drift2, 'A', anomaly_rate, rng)

    X = np.vstack([Xa, Xb, Xc]).astype(np.float32) / 10.0
    y = np.concatenate([ya, yb, yc]).astype(int)

    # İlk eğitim verisi: Kavram A'nın 2000 normal örneği
    normal_idx = np.where(ya == 0)[0][:2000]
    X_init = Xa[normal_idx].astype(np.float32) / 10.0

    print(f"Sea veri seti yüklendi:")
    print(f"  Toplam   : {len(X):,} örnek")
    print(f"  Anomali  : {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  Drift    : t={drift1:,} (Kavram A→B), t={drift2:,} (B→A)")

    return X_init, X, y, [drift1, drift2]


# ══════════════════════════════════════════════════════════════════════
#  CSV DATASET (Kendi Veri Setin)
# ══════════════════════════════════════════════════════════════════════
def load_csv(filepath, label_col=-1, init_size=2000,
             anomaly_rate=None, scale=True, seed=42):
    """
    CSV dosyasından veri yükle.

    Beklenen format:
        - Her satır bir örnek
        - Son sütun (veya label_col) = etiket (0=normal, 1=anomali)
        - Diğer sütunlar = özellikler
        - Başlık satırı olabilir veya olmayabilir

    Parametreler:
        filepath     : CSV dosyasının yolu
        label_col    : etiket sütunu (varsayılan: -1 = son sütun)
        init_size    : ilk eğitim için kullanılacak normal örnek sayısı
        anomaly_rate : None ise veri setindeki orana göre; float verirsen
                       stream'i o orana göre yeniden örnekle
        scale        : True ise StandardScaler uygula
        seed         : rastgelelik için tohum

    Döndürür:
        X_init, X_stream, y_stream, drift_times (boş liste)
    """
    rng = np.random.RandomState(seed)

    # Yükle
    try:
        df = pd.read_csv(filepath, header=0)
    except Exception:
        df = pd.read_csv(filepath, header=None)

    data = df.values.astype(float)
    X = np.delete(data, label_col, axis=1).astype(np.float32)
    y = data[:, label_col].astype(int)

    # Etiketleri 0/1 yap
    unique = np.unique(y)
    if set(unique) != {0, 1}:
        min_class = unique.min()
        y = (y != min_class).astype(int)

    print(f"CSV yüklendi: {filepath}")
    print(f"  Şekil    : {X.shape}")
    print(f"  Anomali  : {y.sum():,} ({y.mean()*100:.2f}%)")

    # Normalizasyon
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
        # Sigmoid girişine uygun hale getir [0,1]
        X = 1 / (1 + np.exp(-X))

    # Anomali oranı ayarla (isteğe bağlı)
    if anomaly_rate is not None:
        X, y = _resample(X, y, anomaly_rate, rng)
        print(f"  Yeniden örneklendi → anomali: {anomaly_rate*100:.1f}%")

    # İlk eğitim verisi (normal örnekler)
    normal_idx = np.where(y == 0)[0]
    rng.shuffle(normal_idx)
    init_idx   = normal_idx[:init_size]
    X_init     = X[init_idx]

    # Stream: geri kalan örnekler
    all_idx    = np.arange(len(X))
    stream_idx = np.setdiff1d(all_idx, init_idx)
    rng.shuffle(stream_idx)
    X_stream   = X[stream_idx]
    y_stream   = y[stream_idx]

    print(f"  İlk eğitim : {len(X_init):,} normal örnek")
    print(f"  Stream     : {len(X_stream):,} örnek")

    return X_init, X_stream, y_stream, []


def _resample(X, y, target_rate, rng):
    """Anomali oranını istenen değere ayarla"""
    normal_idx  = np.where(y == 0)[0]
    anomaly_idx = np.where(y == 1)[0]
    n_normal    = len(normal_idx)
    n_anomaly   = int(n_normal * target_rate / (1 - target_rate))
    n_anomaly   = min(n_anomaly, len(anomaly_idx))

    chosen_anom = rng.choice(anomaly_idx, n_anomaly, replace=False)
    idx = np.concatenate([normal_idx, chosen_anom])
    rng.shuffle(idx)
    return X[idx], y[idx]
