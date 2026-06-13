"""
VAE++ESDD — Veri Seti İndirme Scripti
======================================
Bu scripti kendi bilgisayarında (VS Code terminalinde) çalıştır.

Otomatik (login gerekmez):   MNIST, Forest
Kaggle API ile:              Fraud, Arrhy
Sentetik (indirme yok):      Sea, Sine, Circle, Vib

Kullanım:
  pip install -r requirements.txt
  python download_data.py
"""

import os, sys
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def download_mnist():
    """MNIST — torchvision ile otomatik, login gerekmez"""
    print("\n[1] MNIST indiriliyor...")
    try:
        import torchvision
        t = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: x.view(-1))
        ])
        tr = torchvision.datasets.MNIST(DATA_DIR, train=True,  download=True, transform=t)
        te = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=t)
        X  = np.vstack([tr.data.numpy().reshape(-1,784),
                        te.data.numpy().reshape(-1,784)]).astype(np.float32) / 255.0
        y  = np.concatenate([tr.targets.numpy(), te.targets.numpy()])
        cols = [f'f{i}' for i in range(784)]

        for name, mask, label_fn in [
            ('mnist_01',    np.isin(y,[0,1]), lambda y: (y==1).astype(int)),
            ('mnist_23',    np.isin(y,[2,3]), lambda y: (y==3).astype(int)),
            ('mnist_multi', np.ones(len(y),bool), lambda y: (y!=0).astype(int)),
        ]:
            Xm, ym = X[mask], label_fn(y[mask])
            df = pd.DataFrame(Xm, columns=cols)
            df['label'] = ym
            df.to_csv(os.path.join(DATA_DIR, f'{name}.csv'), index=False)
            print(f"  ✅ {name}.csv  ({len(df):,} örnek | anomali: {ym.sum():,})")
        return True
    except ImportError:
        print("  ❌ pip install torchvision"); return False
    except Exception as e:
        print(f"  ❌ {e}"); return False


def download_forest():
    """Forest/Covertype — ucimlrepo ile otomatik, login gerekmez"""
    path = os.path.join(DATA_DIR, 'forest.csv')
    if os.path.exists(path):
        print(f"\n[2] Forest ⏭️  Zaten mevcut"); return True
    print("\n[2] Forest (Covertype) indiriliyor...")
    try:
        from ucimlrepo import fetch_ucirepo
        ds   = fetch_ucirepo(id=31)
        X    = ds.data.features.values.astype(np.float32)
        y    = ds.data.targets.values.ravel().astype(int)
        ybin = (y > 2).astype(int)
        df   = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
        df['label'] = ybin
        df.to_csv(path, index=False)
        print(f"  ✅ forest.csv ({len(df):,} örnek | anomali: {ybin.mean()*100:.1f}%)")
        return True
    except ImportError:
        print("  ❌ pip install ucimlrepo"); return False
    except Exception as e:
        print(f"  ❌ {e}"); return False


def download_fraud():
    """Credit Card Fraud — Kaggle API"""
    path = os.path.join(DATA_DIR, 'creditcard.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, nrows=5)
        print(f"\n[3] Fraud ⏭️  Zaten mevcut"); return True
    print("\n[3] Fraud (Credit Card) indiriliyor...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'mlg-ulb/creditcardfraud', path=DATA_DIR, unzip=True, quiet=False)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  ✅ creditcard.csv ({len(df):,} örnek)")
            return True
    except Exception:
        pass
    print(f"""
  ❌ Otomatik başarısız → Manuel indir:
  ─────────────────────────────────────────────────────────
  YÖNTEM 1 — Kaggle API (önerilir):
    1. kaggle.com'da ücretsiz hesap aç
    2. Profil → Settings → API → "Create New Token"
    3. kaggle.json dosyasını şuraya koy:
         Windows : C:\\Users\\<KULLANICI>\\.kaggle\\kaggle.json
         Mac/Linux: ~/.kaggle/kaggle.json
    4. Terminalde çalıştır:
         pip install kaggle
         kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip

  YÖNTEM 2 — Tarayıcıdan:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    → Download → ZIP aç → creditcard.csv → data/ klasörüne koy
  ─────────────────────────────────────────────────────────""")
    return False


def download_arrhy():
    """MIT-BIH Arrhythmia — Kaggle API"""
    path = os.path.join(DATA_DIR, 'mitbih_train.csv')
    if os.path.exists(path):
        print(f"\n[4] Arrhy ⏭️  Zaten mevcut"); return True
    print("\n[4] Arrhy (MIT-BIH) indiriliyor...")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'shayanfazeli/heartbeat', path=DATA_DIR, unzip=True, quiet=False)
        if os.path.exists(path):
            df = pd.read_csv(path, header=None)
            print(f"  ✅ mitbih_train.csv ({len(df):,} örnek | 187 özellik)")
            return True
    except Exception:
        pass
    print(f"""
  ❌ Otomatik başarısız → Manuel indir:
  ─────────────────────────────────────────────────────────
  YÖNTEM 1 — Kaggle API:
    kaggle datasets download -d shayanfazeli/heartbeat -p data/ --unzip

  YÖNTEM 2 — Tarayıcıdan:
    https://www.kaggle.com/datasets/shayanfazeli/heartbeat
    → Download → ZIP aç → mitbih_train.csv → data/ klasörüne koy
  ─────────────────────────────────────────────────────────""")
    return False


def main():
    print("=" * 60)
    print("  VAE++ESDD — Dataset İndirme")
    print("  Makale: Neurocomputing 676 (2026)")
    print("=" * 60)
    print("  Sea/Sine/Circle/Vib → kod içinde üretilir, indirme yok\n")

    r = {}
    r['MNIST']  = download_mnist()
    r['Forest'] = download_forest()
    r['Fraud']  = download_fraud()
    r['Arrhy']  = download_arrhy()

    print("\n" + "=" * 60)
    print("  ÖZET")
    print("=" * 60)
    for k, v in r.items():
        print(f"  {'✅' if v else '❌'}  {k}")
    print("  ✅  Sea / Sine / Circle / Vib  (sentetik)")

    print("\n  Hazır dosyalar:")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith('.csv'):
            mb = os.path.getsize(os.path.join(DATA_DIR, f)) / 1024**2
            print(f"    data/{f}  ({mb:.1f} MB)")

    print("\n  Çalıştırma:")
    for ds in ['sea','sine','circle','vib',
               'mnist_01','mnist_23','mnist_multi',
               'forest','fraud','arrhy','all']:
        print(f"    python run.py --dataset {ds}")
    print("=" * 60)


if __name__ == '__main__':
    main()
