Adaptive MLOps Pipeline

Bu proje, telekomünikasyon ağ trafiğindeki değişimleri (Concept Drift) tespit eden, Spark tabanlı hibrit bir MLOps mimarisidir. Sistem, veri akışı sırasında oluşan sapmaları ADWIN algoritması ile algılar ve Kayan Pencere (Sliding Window) stratejisi ile modelini otonom olarak günceller.

⸻

Proje Mimarisi

Sistem aşağıdaki bileşenlerin entegrasyonu ile çalışır:
	1.	ETL & Veri İşleme: Apache Spark (PySpark - Local Mode)
	2.	Model Eğitimi: Scikit-Learn Random Forest (Drift algılandığında tetiklenir)
	3.	İzleme (Monitoring): Grafana, Prometheus Pushgateway ve MinIO

⸻

Proje Dizin Yapısı

SparkProjem/
│
├── README.md                          # Proje Dokümantasyonu
├── requirements.txt                   # Kütüphaneler
├── TUBITAK_2807__030825.csv           # Ana Ham Veri
├── Overlap_matrix.csv                 # Hücre komşuluk verisi
│
├── 📜 pipeline_1_pyspark_robustscaler.py  # Spark 1. Aşama: Veri Hazırlama
├── 📜 pipeline_2_advanced_fs_full.py      # Spark 2. Aşama: PCA & Feature Selection
│
├── 📂 drift_final/                        # Simülasyon Modülü
│   ├── 📂 Dataset/                        # Parçalı batch verileri (batch1.dat...)
│
|── 📂 gas_drift_demo/
│   ├──📜  stream_simulation_files.py      # Canlı Akış ve Drift Tespiti
├── 📂 pipeline1_robust_scaled_pyspark/    # Spark Çıktısı 1 (Parquet)
└── 📂 ml_ready_data_pca_parquet_full/     # Spark Çıktısı 2 (Eğitim Verisi)

⸻

Kurulum ve Çalıştırma Adımları

Python Ortamının Hazırlanması

pip install -r requirements.txt


⸻

 Kafka Altyapısının Başlatılması (Docker Compose)

Kafka ve Zookeeper servislerini başlatmak için:

docker-compose up -d

Bu komut docker-compose.yml dosyasını kullanarak Kafka altyapısını ayağa kaldırır.

⸻

3️⃣ MLOps Servislerinin Başlatılması (Manuel Docker CLI)

 MinIO (S3 Uyumlulu Veri Depolama)

docker run -p 9000:9000 -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"

Prometheus Pushgateway

docker run -p 9091:9091 prom/pushgateway

Grafana

docker run -d -p 3000:3000 --name=grafana grafana/grafana

MLflow Server (Model Takibi - Local)

mlflow server --host 127.0.0.1 --port 5000


⸻

Pipeline’ların Çalıştırılması

Adım 1: Veri Hazırlama (Spark ETL)

Ham veriyi okur, temizler ve RobustScaler uygular.

spark-submit pipeline_1_pyspark_robustscaler.py

Adım 2: Özellik Seçimi ve PCA

Öznitelik mühendisliği yapar ve veriyi boyutsal olarak indirger.

spark-submit pipeline_2_advanced_fs_full.py

Adım 3: Canlı Akış & Drift Simülasyonu

Batch dosyalarını kullanarak canlı veri akışını ve adaptif modeli başlatır.

cd gas_drift_demo
python stream_simulation_files.py


⸻

Canlı İzleme Adresleri

Simülasyon çalışırken sistemi aşağıdaki arayüzlerden izleyebilirsiniz:
	•	Grafana: http://localhost:3000
Kullanıcı: admin / Şifre: admin
	•	MinIO Console: http://localhost:9001
Giriş: minioadmin / minioadmin
	•	Pushgateway Metrics: http://localhost:9091
	•	MLflow UI: http://localhost:5000

⸻
Anahtar Kavramlar
	•	Concept Drift Detection (ADWIN)
	•	Sliding Window Model Update
	•	Streaming MLOps Architecture
	•	Real-time Monitoring & Metrics

⸻

## Results & Progress

**Dataset.** Real LTE cell-performance data from a TUBITAK research project: 100,000+ raw measurement rows across 50 network KPIs collected from real telecom cells, plus a cell-neighborhood overlap matrix for spatial context.

**Pipeline results.** Pipeline 1 (windowing + robust scaling) turns 100K+ raw rows into 4,155 model-ready windowed samples; RobustScaler was chosen because telecom KPI distributions are outlier-heavy. Pipeline 2 (feature selection + PCA) reduces 50 features to 12 components, a 76% dimensionality reduction that preserves 96% of variance and cuts storage from 112 MB to 79 MB. All experiments are tracked as 8 versioned MLflow runs and monitored live on a Grafana + Prometheus dashboard.

**Drift detection results.** The methodology was first validated by replicating Gonzalez-Cebrian et al. (2024) on its real public dataset: the dP index correctly flags known drift between the reference (2005-2010) and current (2010-2015) periods (dP = 34 vs. threshold 30). Across six injected-drift scenarios the two indices proved complementary: dP responds to medium and high scale drift (about 46 and 38 on a 0-100 scale) while dE_PCA responds to mean-shift and added-noise drift, with zero false alarms on the no-drift scenario for both. A VAE at the input layer (TensorFlow) provides reconstruction-based detection alongside ADWIN for streaming drift.

**Problems solved.** High dimensionality of 50 correlated KPIs, solved with the PCA pipeline. Outlier-heavy KPI distributions that break standard scaling, solved with robust scaling in the Spark ETL stage. False-positive drift alarms, solved with the dual-metric dP + dE_PCA strategy, with dE_PCA preferred for its false-alarm resistance. Reproducibility, solved with MLflow-tracked, versioned experiments.

**Status.** Data preparation, drift detection and monitoring layers are complete and verified. The drift-adaptation stage, sliding-window model retraining and automated model updates, is in final development.

Bu proje, akademik çalışmalar ve gerçek zamanlı ağ trafiği analizi için uçtan uca adaptif bir MLOps örneği sunar.
