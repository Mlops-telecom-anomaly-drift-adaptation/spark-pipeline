# TUBITAK Telecommunications MLOps Project

## Proje Hakkında

Bu proje, LTE hücre performans verilerini işlemek ve analiz etmek için geliştirilmiş bir MLOps pipeline'ıdır. Apache Spark kullanılarak 100,000+ satır veriyi işler, MLflow ile deney takibi yapar ve Prometheus/Grafana ile sistem monitörleme sağlar.

## Özellikler

- **İki Aşamalı Spark Pipeline**: LTE hücre verilerinin verimli işlenmesi
- **PCA Boyut İndirgeme**: Yüksek boyutlu verilerin optimize edilmesi
- **MLflow Entegrasyonu**: Deney takibi ve model versiyonlama
- **Prometheus & Grafana**: Gerçek zamanlı sistem monitörleme
- **Ölçeklenebilir Mimari**: Büyük veri setleri için optimize edilmiş

## Teknolojiler

- **Apache Spark**: Dağıtık veri işleme
- **MLflow**: ML deney yönetimi ve tracking
- **Prometheus**: Metrik toplama ve monitoring
- **Grafana**: Görselleştirme ve dashboard
- **Python**: Ana programlama dili
- **PCA**: Dimensionality reduction

## Kurulum

### Gereksinimler

```bash
# Python bağımlılıkları
pip install -r requirements.txt
```

### requirements.txt
```
pyspark==3.4.0
mlflow==2.8.0
prometheus-client==0.18.0
pandas
numpy
scikit-learn
```

### Yapılandırma

1. MLflow tracking server'ı başlatın:
```bash
mlflow server --host 0.0.0.0 --port 5000
```

2. Prometheus yapılandırmasını ayarlayın (`prometheus.yml`)

3. Grafana dashboard'larını import edin

## Kullanım

### Veri İşleme Pipeline'ı Çalıştırma

```python
# Pipeline'ı başlat
python src/main_pipeline.py --input data/lte_cell_data.csv --output results/
```

### MLflow Deneyleri

```python
# MLflow UI'a erişim
mlflow ui --port 5000
```

Tarayıcınızda `http://localhost:5000` adresine gidin.

## Proje Yapısı

```
.
├── data/                   # Veri dosyaları
│   └── lte_cell_data.csv
├── src/                    # Kaynak kodlar
│   ├── pipeline/
│   │   ├── stage1.py      # İlk aşama pipeline
│   │   └── stage2.py      # İkinci aşama pipeline
│   ├── preprocessing/
│   │   └── pca.py         # PCA işlemleri
│   └── main_pipeline.py   # Ana çalıştırma scripti
├── notebooks/              # Jupyter notebooks
├── config/                 # Yapılandırma dosyaları
│   ├── prometheus.yml
│   └── grafana_dashboard.json
├── docs/                   # Dökümantasyon
├── tests/                  # Test dosyaları
├── requirements.txt
└── README.md
```

## Pipeline Aşamaları

### Aşama 1: Veri Ön İşleme
- Ham LTE hücre verilerinin yüklenmesi
- Veri temizleme ve normalizasyon
- Eksik değerlerin doldurulması
- Feature engineering

### Aşama 2: Model Eğitimi ve PCA
- PCA ile boyut indirgeme
- Model eğitimi
- MLflow ile metrik kaydetme
- Model kaydetme ve versiyonlama

## Monitoring

### Prometheus Metrikleri
- Pipeline çalışma süreleri
- Veri işleme throughput
- Hata oranları
- Kaynak kullanımı (CPU, Memory)

### Grafana Dashboards
- Real-time pipeline monitoring
- Model performans metrikleri
- Sistem sağlık göstergeleri

## Katkıda Bulunanlar

- **Sena** - Geliştirici
- **Zeynep** - Geliştirici

## Lisans

TUBITAK projesi kapsamında geliştirilmiştir.

## İletişim

Proje hakkında sorularınız için GitHub Issues kullanabilirsiniz.

## Notlar

- Veri seti 100,000+ satır LTE hücre performans verisi içermektedir
- Pipeline optimize edilmiş Spark konfigürasyonu ile çalışır
- MLflow tracking URI'ı `config/mlflow_config.py` dosyasında ayarlanabilir

## Gelecek Geliştirmeler

- [ ] Otomatik model retraining
- [ ] A/B testing framework
- [ ] Model deployment automation
- [ ] Advanced feature engineering
- [ ] Real-time prediction API

---

**Son Güncelleme**: Kasım 2025
