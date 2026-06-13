"""
MLOps Pipeline — VAE++ESDD ile Drift Tespiti
=============================================
Araçlar:
    Spark    → veri işleme (batch)
    VAE      → anomali tespiti
    MLflow   → model ve metrik takibi
    Prometheus → metrik servisi
    Grafana  → dashboard (Prometheus'u okur)

Çalıştırma:
    python mlops_pipeline.py

Grafana dashboard için:
    1. Prometheus kur: https://prometheus.io/download/
    2. prometheus.yml dosyasına ekle:
         scrape_configs:
           - job_name: 'vae_drift'
             static_configs:
               - targets: ['localhost:8000']
    3. Grafana'da Prometheus datasource ekle
    4. Dashboard'da metrikleri izle
"""

import os
import sys
import time
import threading
import numpy as np
import mlflow
import mlflow.pytorch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_all   import load_sea
from src.model      import VAEplusESDD
from src.evaluation import PrequentialEvaluator

# ── Prometheus ────────────────────────────────────────────────────────
try:
    from prometheus_client import start_http_server, Gauge, Counter
    PROMETHEUS_OK = True
except ImportError:
    print("prometheus_client bulunamadı → pip install prometheus-client")
    PROMETHEUS_OK = False

# ── Spark ─────────────────────────────────────────────────────────────
try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    SPARK_OK = True
except Exception:
    SPARK_OK = False


# ══════════════════════════════════════════════════════════════════════
#  PROMETHEUS METRİKLERİ
# ══════════════════════════════════════════════════════════════════════
if PROMETHEUS_OK:
    GAUGE_GMEAN       = Gauge('vae_gmean',       'G-mean skoru')
    GAUGE_RECALL      = Gauge('vae_recall',      'Recall skoru')
    GAUGE_SPEC        = Gauge('vae_specificity', 'Specificity skoru')
    GAUGE_PAUC        = Gauge('vae_pauc',        'PAUC skoru')
    GAUGE_DRIFT_COUNT = Gauge('vae_drift_count', 'Toplam drift alarm sayısı')
    GAUGE_ANOMALY_RATE= Gauge('vae_anomaly_rate','Anlık anomali oranı')
    COUNTER_PROCESSED = Counter('vae_processed_total', 'İşlenen örnek sayısı')


# ══════════════════════════════════════════════════════════════════════
#  SPARK: VERİ HAZIRLAMA
# ══════════════════════════════════════════════════════════════════════
def create_spark_dataframe(X, y):
    """
    Numpy array → Spark DataFrame
    Spark burada veriyi batch olarak işliyor.
    Gerçek senaryoda: Kafka, HDFS veya S3'ten okur.
    """
    if not SPARK_OK:
        return None, None

    try:
        spark = SparkSession.builder \
            .appName("VAE_ESDD_DriftDetection") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
    except Exception:
        print("  Spark başlatılamadı, numpy ile devam ediliyor.")
        return None, None

    # Numpy → Pandas → Spark
    import pandas as pd
    df_pd = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    df_pd['label'] = y
    df_spark = spark.createDataFrame(df_pd)

    print(f"  Spark DataFrame: {df_spark.count():,} satır, "
          f"{len(df_spark.columns)} sütun")

    return spark, df_spark


def spark_batch_to_numpy(df_spark, batch_size=500):
    """
    Spark DataFrame'i batch'lere böl.
    Her batch bir zaman dilimini temsil eder.
    """
    df_pd    = df_spark.toPandas()
    features = [c for c in df_pd.columns if c != 'label']
    X = df_pd[features].values.astype('float32')
    y = df_pd['label'].values.astype(int)

    batches = []
    for i in range(0, len(X), batch_size):
        batches.append((X[i:i+batch_size], y[i:i+batch_size]))
    return batches


# ══════════════════════════════════════════════════════════════════════
#  ANA PIPELINE
# ══════════════════════════════════════════════════════════════════════
def run_pipeline():

    print("=" * 60)
    print("  MLOps Pipeline — VAE++ESDD")
    print("  Spark + MLflow + Prometheus + Grafana")
    print("=" * 60)

    # ── 1. Prometheus başlat ─────────────────────────────────────────
    if PROMETHEUS_OK:
        start_http_server(8000)
        print("\nPrometheus metrikleri → http://localhost:8000")
        print("Grafana'da bu adresi datasource olarak ekle.\n")

    # ── 2. MLflow experiment ─────────────────────────────────────────
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("VAE_ESDD_DriftDetection")

    with mlflow.start_run(run_name="sea_dataset_batch"):

        # Parametreleri logla
        mlflow.log_params({
            "dataset"      : "sea",
            "anomaly_rate" : 0.01,
            "n_ensemble"   : 10,
            "w_train"      : 3000,
            "beta"         : 1.0,
            "p_alarm"      : 0.001,
        })

        # ── 3. Veri üret ─────────────────────────────────────────────
        print("[1] Sea verisi üretiliyor...")
        X_init, X_stream, y_stream, drift_times = load_sea(
            anomaly_rate=0.01, seed=42)
        print(f"    Stream: {len(X_stream):,} örnek")
        print(f"    Drift zamanları: {drift_times}")

        # ── 4. Spark DataFrame ────────────────────────────────────────
        print("\n[2] Spark DataFrame oluşturuluyor...")
        spark, df_spark = create_spark_dataframe(X_stream, y_stream)

        if spark is not None:
            batches = spark_batch_to_numpy(df_spark, batch_size=500)
            print(f"    {len(batches)} batch hazır (500 örnek/batch)")
        else:
            # Spark yoksa direkt numpy kullan
            batch_size = 500
            batches = [(X_stream[i:i+batch_size], y_stream[i:i+batch_size])
                       for i in range(0, len(X_stream), batch_size)]

        # ── 5. Model ─────────────────────────────────────────────────
        print("\n[3] VAE++ESDD modeli başlatılıyor...")
        model = VAEplusESDD(
            input_dim=2, n=10, W_train=3000, gamma=2000,
            W_drift_min=180, W_drift_max=220,
            P_thre=1, D_thre=10,
            P_warn=0.01, P_alarm=0.001,
            beta=1.0, lr=0.001, num_epochs=10,
        )
        model.initialize(X_init)

        # ── 6. Batch stream işleme ────────────────────────────────────
        print("\n[4] Batch stream işleniyor...\n")
        evaluator  = PrequentialEvaluator(fading=0.99, pauc_window=1000)
        step       = 0
        log_every  = 10  # Her 10 batch'te bir MLflow'a logla

        for batch_idx, (X_batch, y_batch) in enumerate(batches):
            batch_preds   = []
            batch_anomaly = 0

            # Batch içindeki her örneği işle
            for xi, yi in zip(X_batch, y_batch):
                pred, score, drift_alarm = model.process(xi)
                evaluator.update(int(yi), pred, score)
                batch_preds.append(pred)
                batch_anomaly += pred
                step += 1

                if PROMETHEUS_OK:
                    COUNTER_PROCESSED.inc()

            # Batch metrikleri hesapla
            gmean  = np.mean(evaluator.gmean_hist[-500:]) if evaluator.gmean_hist else 0
            recall = np.mean(evaluator.recall_hist[-500:]) if evaluator.recall_hist else 0
            spec   = np.mean(evaluator.spec_hist[-500:])   if evaluator.spec_hist   else 0
            pauc   = np.mean(evaluator.pauc_hist[-500:])   if evaluator.pauc_hist   else 0
            anom_rate = batch_anomaly / len(X_batch)

            # ── Prometheus güncelle ───────────────────────────────────
            if PROMETHEUS_OK:
                GAUGE_GMEAN.set(gmean)
                GAUGE_RECALL.set(recall)
                GAUGE_SPEC.set(spec)
                GAUGE_PAUC.set(pauc)
                GAUGE_DRIFT_COUNT.set(len(model.drift_alarms))
                GAUGE_ANOMALY_RATE.set(anom_rate)

            # ── MLflow'a logla ────────────────────────────────────────
            if batch_idx % log_every == 0:
                mlflow.log_metrics({
                    "gmean"       : round(gmean,  4),
                    "recall"      : round(recall, 4),
                    "specificity" : round(spec,   4),
                    "pauc"        : round(pauc,   4),
                    "drift_count" : len(model.drift_alarms),
                    "anomaly_rate": round(anom_rate, 4),
                }, step=step)

            # ── Terminal log ──────────────────────────────────────────
            print(f"  Batch {batch_idx+1:>3}/{len(batches)} "
                  f"| t={step:>6,} "
                  f"| G-mean={gmean:.3f} "
                  f"| Recall={recall:.3f} "
                  f"| Drift={len(model.drift_alarms)}")

        # ── 7. Final metrikler ────────────────────────────────────────
        metrics = evaluator.summary()
        print(f"\n{'='*60}")
        print("  SONUÇLAR")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"  {k:<16}: {v:.4f}")
        print(f"  Drift alarmları : {len(model.drift_alarms)}")
        print(f"{'='*60}")

        # MLflow final
        mlflow.log_metrics({
            "final_gmean"      : metrics['G-mean'],
            "final_recall"     : metrics['Recall'],
            "final_specificity": metrics['Specificity'],
            "final_pauc"       : metrics['PAUC'],
            "total_drifts"     : len(model.drift_alarms),
        })

        mlflow.set_tags({
            "drift_times"  : str(drift_times),
            "drift_alarms" : str(model.drift_alarms[:5]) + "...",
        })

        print("\nMLflow UI için:")
        print("  mlflow ui")
        print("  → http://localhost:5000")

        # ── 8. Prometheus açık kalsın ─────────────────────────────────
        if PROMETHEUS_OK:
            print("\nPrometheus metrikleri açık: http://localhost:8000")
            print("Durdurmak için Ctrl+C\n")
            try:
                while True:
                    time.sleep(5)
            except KeyboardInterrupt:
                print("\nDurduruldu.")

        if spark is not None:
            spark.stop()


if __name__ == '__main__':
    run_pipeline()
