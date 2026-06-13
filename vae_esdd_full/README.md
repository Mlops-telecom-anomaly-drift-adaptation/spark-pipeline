# VAE++ESDD — Tam Kurulum ve Çalıştırma Rehberi

**Makale:** Drift-aware variational autoencoder-based anomaly detection with two-level ensembling  
**Kaynak kodu:** https://github.com/Jin000001/VAEESDD  
**Yayın:** Neurocomputing 676 (2026)

---

## ⚡ Hızlı Başlangıç (5 dakika)

```bash
# 1. Kütüphaneleri kur
pip install -r requirements.txt

# 2. Datasetleri indir
python download_data.py

# 3. Çalıştır
python run.py --dataset sea      # sentetik, hemen çalışır
python run.py --dataset all      # tüm datasetler
```

---

## 📦 Dataset İndirme Rehberi

### Otomatik (login gerekmez)

| Dataset | Komut | Boyut |
|---------|-------|-------|
| MNIST (01/23/multi) | `python download_data.py` | ~55 MB |
| Forest (Covertype) | `python download_data.py` | ~70 MB |
| Sea/Sine/Circle/Vib | Otomatik üretilir | — |

### Kaggle ile (ücretsiz, 2 dakika)

**Adım 1 — Kaggle API Key Al:**
1. https://www.kaggle.com → Ücretsiz hesap aç
2. Sağ üst → Profil resmi → **Settings**
3. **API** bölümü → **Create New Token** → `kaggle.json` indirilir
4. Dosyayı şuraya koy:
   - Windows: `C:\Users\KULLANICI_ADIN\.kaggle\kaggle.json`
   - Mac/Linux: `~/.kaggle/kaggle.json`

**Adım 2 — Paket Kur:**
```bash
pip install kaggle
```

**Adım 3 — İndir:**
```bash
# Fraud (Credit Card)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip

# Arrhy (MIT-BIH ECG)
kaggle datasets download -d shayanfazeli/heartbeat -p data/ --unzip
```

### Manuel İndirme (Kaggle API olmadan)

Tarayıcıdan indir, `data/` klasörüne koy:

| Dataset | Link | Kaydedilecek dosya |
|---------|------|--------------------|
| Fraud | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | `data/creditcard.csv` |
| Arrhy | https://www.kaggle.com/datasets/shayanfazeli/heartbeat | `data/mitbih_train.csv` |

---

## 🚀 Çalıştırma

```bash
# Tek dataset
python run.py --dataset sea
python run.py --dataset sine
python run.py --dataset circle
python run.py --dataset vib
python run.py --dataset mnist_01
python run.py --dataset mnist_23
python run.py --dataset mnist_multi
python run.py --dataset forest
python run.py --dataset fraud
python run.py --dataset arrhy

# Tümü (makale Tablo 6 gibi)
python run.py --dataset all

# Anomali oranı %0.1 (extreme imbalance)
python run.py --dataset sea --anomaly_rate 0.001
```

---

## 📁 Proje Yapısı

```
vae_esdd_project/
├── run.py              ← Ana çalıştırma dosyası
├── download_data.py    ← Dataset indirme
├── requirements.txt    ← Kütüphaneler
├── README.md
├── data/               ← Datasetler buraya
│   ├── mnist_01.csv
│   ├── mnist_23.csv
│   ├── mnist_multi.csv
│   ├── forest.csv
│   ├── creditcard.csv
│   └── mitbih_train.csv
├── results/            ← Çıktılar
│   ├── results.png
│   ├── summary.csv
│   └── stream_log.csv
└── src/
    ├── vae.py          ← VAE (Autoencoder mimarisi)
    ├── detector.py     ← Mann-Whitney drift dedektörü
    ├── model.py        ← VAE++ESDD Algorithm 1
    ├── data_all.py     ← Tüm dataset yükleyicileri
    └── evaluation.py   ← G-mean, PAUC, Recall, Specificity
```

---

## 📊 Makale Parametreleri (Tablo 3)

| Parametre | Sea | Forest | MNIST | Fraud | Arrhy |
|-----------|-----|--------|-------|-------|-------|
| Gizli katmanlar | [64,8] | [64,32] | [512,256,64] | [64,32,8] | [128,32] |
| Öğrenme hızı | 0.001 | 0.0001 | 0.0001 | 0.001 | 0.0001 |
| Loss | BCE | MSE | BCE | MSE | MSE |
| Epoch | 10 | 50 | 10 | 10 | 10 |
| Beta (KL) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Ensemble n | 10 | 10 | 10 | 10 | 10 |

---

## 📈 Beklenen Sonuçlar (Makale Tablo 6 — G-mean)

| Dataset | VAE++ESDD | StrAEm++DD |
|---------|-----------|------------|
| Sea | **0.868** | 0.790 |
| Sine | **0.791** | 0.704 |
| Circle | **0.760** | 0.523 |
| Forest | **0.817** | 0.627 |
| Fraud | **0.821** | 0.797 |
| MNIST-01 | **0.744** | 0.726 |
