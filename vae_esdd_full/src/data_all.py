# src/data_all.py
"""
Tüm Dataset Yükleyicileri (Makale Tablo 2)

Her fonksiyon döndürür:
    X_init   (np.ndarray) : ilk eğitim için normal örnekler
    X_stream (np.ndarray) : stream verisi
    y_stream (np.ndarray) : etiketler (0=normal, 1=anomali)
    drift_times (list)    : drift zamanları
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

INIT_SIZE = 2000  # ilk eğitim için normal örnek sayısı


# ── Yardımcılar ──────────────────────────────────────────────────────

def _imbalance(X, y, rate, rng):
    """Anomali oranını ayarla"""
    n0 = (y == 0).sum()
    n1 = max(1, int(n0 * rate / (1 - rate)))
    n1 = min(n1, (y == 1).sum())
    i0 = rng.choice(np.where(y==0)[0], n0,  replace=False)
    i1 = rng.choice(np.where(y==1)[0], n1,  replace=False)
    idx = np.concatenate([i0, i1])
    rng.shuffle(idx)
    return X[idx], y[idx]

def _split(X, y, init_size, rng):
    """Normal örneklerden başlangıç verisi al"""
    ni  = np.where(y == 0)[0]
    rng.shuffle(ni)
    iidx = ni[:init_size]
    mask = np.ones(len(X), bool); mask[iidx] = False
    return X[iidx], X[mask], y[mask]

def _sigmoid_norm(X):
    """StandardScaler + Sigmoid → [0,1]"""
    X = StandardScaler().fit_transform(X)
    return (1 / (1 + np.exp(-X))).astype(np.float32)

def _load_csv(filename, label_col='label'):
    """CSV yükle"""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{filename}' bulunamadı.\n"
            f"→ python download_data.py komutunu çalıştır.")
    df = pd.read_csv(path)
    y  = df[label_col].values.astype(int)
    X  = df.drop(label_col, axis=1).values.astype(np.float32)
    return X, y


# ══════════════════════════════════════════════════════════════════════
#  SENTETİK DATASETLER
# ══════════════════════════════════════════════════════════════════════

def load_sea(anomaly_rate=0.01, n=20000, seed=42):
    """Recurrent drift: t=10000 (A→B), t=15000 (B→A)"""
    rng = np.random.RandomState(seed)
    d1, d2 = 10000, 15000

    def seg(size, concept, rate, rng):
        na = max(1, int(size*rate)); nn = size-na
        pts = rng.uniform(0, 10, (size*30, 2))
        nm  = (pts[:,0]+pts[:,1]>7) if concept=='A' else (pts[:,0]+pts[:,1]<=7)
        X = np.vstack([pts[nm][:nn], pts[~nm][:na]]).astype(np.float32)/10
        y = np.concatenate([np.zeros(nn), np.ones(na)]).astype(int)
        idx = rng.permutation(len(X)); return X[idx], y[idx]

    Xa,ya = seg(d1,'A',anomaly_rate,rng)
    Xb,yb = seg(d2-d1,'B',anomaly_rate,rng)
    Xc,yc = seg(n-d2,'A',anomaly_rate,rng)
    X = np.vstack([Xa,Xb,Xc]); y = np.concatenate([ya,yb,yc])
    Xi = Xa[np.where(ya==0)[0][:INIT_SIZE]]
    mask = np.ones(len(X),bool)
    mask[np.where(ya==0)[0][:INIT_SIZE]] = False
    print(f"Sea: {len(X):,} örnek | anomali {y.mean()*100:.1f}% | drift=[{d1},{d2}]")
    return Xi, X[mask], y[mask], [d1, d2]


def load_sine(anomaly_rate=0.01, n=10000, seed=42):
    """Abrupt drift: t=5000"""
    rng = np.random.RandomState(seed); d = n//2

    def seg(size, concept, rate, rng):
        na = max(1,int(size*rate)); nn = size-na
        pts = rng.uniform([0,-1],[np.pi,1],(size*20,2))
        sin = np.sin(pts[:,0])
        nm  = (pts[:,1]>sin) if concept=='A' else (pts[:,1]<sin)
        X = np.vstack([pts[nm][:nn], pts[~nm][:na]]).astype(np.float32)
        X[:,0] /= np.pi; X[:,1] = (X[:,1]+1)/2
        y = np.concatenate([np.zeros(nn),np.ones(na)]).astype(int)
        idx = rng.permutation(len(X)); return X[idx], y[idx]

    Xa,ya = seg(d,'A',anomaly_rate,rng)
    Xb,yb = seg(n-d,'B',anomaly_rate,rng)
    X = np.vstack([Xa,Xb]); y = np.concatenate([ya,yb])
    Xi, Xs, ys = _split(X, y, INIT_SIZE, rng)
    print(f"Sine: {len(X):,} örnek | anomali {y.mean()*100:.1f}% | drift=[{d}]")
    return Xi, Xs, ys, [d]


def load_circle(anomaly_rate=0.01, n=10000, seed=42):
    """Abrupt drift: t=5000"""
    rng = np.random.RandomState(seed); d = n//2
    cx, cy, r = 0.4, 0.5, 0.2

    def seg(size, concept, rate, rng):
        na = max(1,int(size*rate)); nn = size-na
        pts = rng.uniform(0,1,(size*20,2))
        ins = np.sqrt((pts[:,0]-cx)**2+(pts[:,1]-cy)**2)<r
        nm  = ins if concept=='A' else ~ins
        X = np.vstack([pts[nm][:nn], pts[~nm][:na]]).astype(np.float32)
        y = np.concatenate([np.zeros(nn),np.ones(na)]).astype(int)
        idx = rng.permutation(len(X)); return X[idx], y[idx]

    Xa,ya = seg(d,'A',anomaly_rate,rng)
    Xb,yb = seg(n-d,'B',anomaly_rate,rng)
    X = np.vstack([Xa,Xb]); y = np.concatenate([ya,yb])
    Xi, Xs, ys = _split(X, y, INIT_SIZE, rng)
    print(f"Circle: {len(X):,} örnek | anomali {y.mean()*100:.1f}% | drift=[{d}]")
    return Xi, Xs, ys, [d]


def load_vib(anomaly_rate=0.01, n=10000, seed=42):
    """Incremental drift: Normal N(0,1)→N(3,1) kademeli"""
    rng = np.random.RandomState(seed); d = n//2
    nf  = 10

    def seg(size, mu_n, rate, rng):
        na = max(1,int(size*rate)); nn = size-na
        Xn = rng.normal(mu_n, 1, (nn,nf)).astype(np.float32)
        Xa = rng.normal(5,   1, (na,nf)).astype(np.float32)
        X  = np.vstack([Xn,Xa])
        y  = np.concatenate([np.zeros(nn),np.ones(na)]).astype(int)
        idx = rng.permutation(len(X)); return X[idx], y[idx]

    Xa,ya = seg(d,0,anomaly_rate,rng)
    steps = 10; ssz = (n-d)//steps
    Xbs,ybs = [],[]
    for i in range(steps):
        mu = 3*(i+1)/steps
        Xs,ys = seg(ssz,mu,anomaly_rate,rng)
        Xbs.append(Xs); ybs.append(ys)
    Xb,yb = np.vstack(Xbs), np.concatenate(ybs)
    X = _sigmoid_norm(np.vstack([Xa,Xb]))
    y = np.concatenate([ya,yb])
    Xi, Xs, ys = _split(X, y, INIT_SIZE, rng)
    print(f"Vib: {len(X):,} örnek | anomali {y.mean()*100:.1f}% | incremental drift")
    return Xi, Xs, ys, [d]


# ══════════════════════════════════════════════════════════════════════
#  GERÇEK DÜNYA DATASETLER
# ══════════════════════════════════════════════════════════════════════

def load_mnist_01(anomaly_rate=0.01, seed=42):
    """MNIST-01: Normal=0, Anomali=1 | Abrupt drift: t=2500"""
    rng = np.random.RandomState(seed)
    X, y = _load_csv('mnist_01.csv')
    X, y = _imbalance(X, y, anomaly_rate, rng)
    # Drift simülasyonu: hafif görüntü kaydırma (Tablo 2: width/height ±10%)
    half = len(X)//2
    X[half:] = np.roll(X[half:].reshape(-1,28,28), 2, axis=1).reshape(-1,784)
    Xi, Xs, ys = _split(X, y, INIT_SIZE, rng)
    print(f"MNIST-01: {len(X):,} örnek | anomali {y.mean()*100:.1f}%")
    return Xi, Xs, ys, [2500]


def load_mnist_23(anomaly_rate=0.01, seed=42):
    """MNIST-23: Normal=2, Anomali=3 | Abrupt drift t=2500"""
    rng = np.random.RandomState(seed)
    X, y = _load_csv('mnist_23.csv')
    X, y = _imbalance(X, y, anomaly_rate, rng)
    Xi, Xs, ys = _split(X, y, INIT_SIZE, rng)
    print(f"MNIST-23: {len(X):,} örnek | anomali {y.mean()*100:.1f}%")
    return Xi, Xs, ys, [2500]


def load_mnist_multi(anomaly_rate=0.01, seed=42):
    """MNIST-multi: Normal=0, Anomali=1-9"""
    rng = np.random.RandomState(seed)
    X, y = _load_csv('mnist_multi.csv')
    X, y = _imbalance(X, y, anomaly_rate, rng)
    Xi, Xs, ys = _split(X, y, INIT_SIZE, rng)
    print(f"MNIST-multi: {len(X):,} örnek | anomali {y.mean()*100:.1f}%")
    return Xi, Xs, ys, []


def load_forest(anomaly_rate=0.01, seed=42):
    """Forest Covertype | Abrupt drift t=50000"""
    rng = np.random.RandomState(seed)
    X, y = _load_csv('forest.csv')
    X, y = _imbalance(X, y, anomaly_rate, rng)
    X    = _sigmoid_norm(X)
    Xi, Xs, ys = _split(X, y, INIT_SIZE, rng)
    print(f"Forest: {len(X):,} örnek | anomali {y.mean()*100:.1f}%")
    return Xi, Xs, ys, [50000]


def load_fraud(anomaly_rate=0.01, seed=42):
    """Credit Card Fraud | Recurrent drift"""
    rng  = np.random.RandomState(seed)
    path = os.path.join(DATA_DIR, 'creditcard.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            "creditcard.csv bulunamadı\n"
            "→ python download_data.py çalıştır veya README.md'ye bak")
    df = pd.read_csv(path)
    # Son sütun 'Class', Time sütununu çıkar
    X  = df.drop(['Time','Class'], axis=1, errors='ignore').values.astype(np.float32)
    y  = df['Class'].values.astype(int)
    X, y = _imbalance(X, y, anomaly_rate, rng)
    X    = _sigmoid_norm(X)
    # Recurrent drift simülasyonu: 5000. ve 10000. adımda özellikler ×0.1
    n = len(X)
    for d in [n//3, 2*n//3]:
        X[d:min(d+n//6, n)] *= 0.1
    Xi, Xs, ys = _split(X, y, INIT_SIZE, rng)
    print(f"Fraud: {len(X):,} örnek | anomali {y.mean()*100:.1f}%")
    return Xi, Xs, ys, [n//3, 2*n//3]


def load_arrhy(anomaly_rate=0.01, seed=42):
    """MIT-BIH Arrhythmia | 187 özellik, ~80K örnek"""
    rng  = np.random.RandomState(seed)
    path = os.path.join(DATA_DIR, 'mitbih_train.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            "mitbih_train.csv bulunamadı\n"
            "→ python download_data.py çalıştır veya README.md'ye bak")
    df = pd.read_csv(path, header=None)
    X  = df.iloc[:, :187].values.astype(np.float32)
    y_raw = df.iloc[:, 187].values.astype(int)
    # 0=Normal, 1-4=Arrhythmia türleri → binary
    y = (y_raw > 0).astype(int)
    X, y = _imbalance(X, y, anomaly_rate, rng)
    Xi, Xs, ys = _split(X, y, INIT_SIZE, rng)
    print(f"Arrhy: {len(X):,} örnek | anomali {y.mean()*100:.1f}%")
    return Xi, Xs, ys, []


# ── Dataset map ──────────────────────────────────────────────────────
DATASET_MAP = {
    'sea'         : load_sea,
    'sine'        : load_sine,
    'circle'      : load_circle,
    'vib'         : load_vib,
    'mnist_01'    : load_mnist_01,
    'mnist_23'    : load_mnist_23,
    'mnist_multi' : load_mnist_multi,
    'forest'      : load_forest,
    'fraud'       : load_fraud,
    'arrhy'       : load_arrhy,
}

# Makale Tablo 3: Her dataset için model parametreleri
DATASET_PARAMS = {
    'sea'         : {'hidden_dims':(64,8),      'latent_dim':2,  'lr':0.001,  'loss':'bce', 'epochs':10},
    'sine'        : {'hidden_dims':(8,),        'latent_dim':2,  'lr':0.001,  'loss':'bce', 'epochs':10},
    'circle'      : {'hidden_dims':(8,),        'latent_dim':2,  'lr':0.001,  'loss':'bce', 'epochs':50},
    'vib'         : {'hidden_dims':(64,32),     'latent_dim':8,  'lr':0.001,  'loss':'bce', 'epochs':10},
    'mnist_01'    : {'hidden_dims':(512,256),   'latent_dim':64, 'lr':0.0001, 'loss':'bce', 'epochs':10},
    'mnist_23'    : {'hidden_dims':(512,256),   'latent_dim':64, 'lr':0.0001, 'loss':'bce', 'epochs':10},
    'mnist_multi' : {'hidden_dims':(512,256),   'latent_dim':64, 'lr':0.0001, 'loss':'bce', 'epochs':10},
    'forest'      : {'hidden_dims':(64,32),     'latent_dim':16, 'lr':0.0001, 'loss':'mse', 'epochs':50},
    'fraud'       : {'hidden_dims':(64,32,8),   'latent_dim':4,  'lr':0.001,  'loss':'mse', 'epochs':10},
    'arrhy'       : {'hidden_dims':(128,32),    'latent_dim':8,  'lr':0.0001, 'loss':'mse', 'epochs':10},
}
