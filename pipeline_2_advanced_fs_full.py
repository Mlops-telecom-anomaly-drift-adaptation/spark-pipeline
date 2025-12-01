import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import PCA, VarianceThresholdSelector
import mlflow
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# --- 1. AYARLAR ---
# Pipeline 1 (PySpark) çıktısını okuyoruz (Klasör adı)
ARA_GIRDI_DOSYASI = "pipeline1_robust_scaled_pyspark"
SON_CIKTI_DOSYASI = "ml_ready_data_pca_parquet_full" 

PCA_K_DEGERI = 2

# --- 2. MLflow & MinIO ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pipeline 2 - Advanced Features (PySpark)")

# --- 3. Prometheus ---
PUSHGATEWAY_ADDRESS = 'localhost:9091'
PROMETHEUS_JOB_NAME = 'ml_advanced_fs_monitoring_full' 
registry = CollectorRegistry()

# Grafana Metrikleri
g_job_duration = Gauge('spark_adv_fs_job_duration_seconds', 'Islem Suresi', registry=registry)
g_avg_traffic = Gauge('spark_adv_fs_avg_traffic_8h', 'Drift Trafik', registry=registry)
g_avg_drop = Gauge('spark_adv_fs_avg_erab_drop_rate_8h', 'Drift Dusme Orani', registry=registry)
g_pca_features = Gauge('spark_adv_fs_pca_features_count', 'PCA Boyut', registry=registry)

# --- 4. BAŞLAT ---
print(f"PIPELINE 2 (PySpark Integration) başlatılıyor...")
start_time = time.time()

# Zaman formatı hatası almamak için ayarlar eklendi
spark = SparkSession.builder \
    .appName("Pipeline2_Advanced_FS") \
    .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
    .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED") \
    .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
    .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED") \
    .getOrCreate()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    
    try:
        # --- 5. VERİ OKUMA ---
        print(f"Veri okunuyor: {ARA_GIRDI_DOSYASI}")
        if not os.path.exists(ARA_GIRDI_DOSYASI):
            raise Exception(f"HATA: Girdi dosyası bulunamadı! Lütfen önce Pipeline 1'i çalıştırın.")
            
        df_windowed = spark.read.parquet(ARA_GIRDI_DOSYASI)
        print(f"Veri yüklendi. Satır: {df_windowed.count()}")
        
        # Sütun isimlerini kontrol et (Büyük/Küçük harf uyumu için)
        print("Sütunlar:", df_windowed.columns)

        # --- 6. DATA DRIFT ---
        print("Data Drift hesaplanıyor...")
        
        # Pipeline 1'in ürettiği tam sütun adlarını kullanıyoruz
        # (Genellikle: avg_Traffic_Volume_Gbyte_8h gibi olur)
        drift_traffic = 0
        drift_drop = 0
        
        # Sütun adını dinamik bulma (Büyük/küçük harf riskine karşı)
        traffic_col = next((c for c in df_windowed.columns if "Traffic_Volume" in c), None)
        drop_col = next((c for c in df_windowed.columns if "ERAB_Drop" in c), None)
        
        if traffic_col:
            drift_traffic = df_windowed.agg(avg(traffic_col)).first()[0] or 0
            print(f"Trafik Metriği ({traffic_col}): {drift_traffic}")
        
        if drop_col:
            drift_drop = df_windowed.agg(avg(drop_col)).first()[0] or 0
            print(f"Düşme Metriği ({drop_col}): {drift_drop}")

        # --- 7. PCA VE FEATURE SELECTION ---
        print("Öznitelik işlemleri (PCA) yapılıyor...")
        
        # Pipeline 1 zaten 'scaled_features' adında vektör üretti. Onu kullanıyoruz.
        if "scaled_features" not in df_windowed.columns:
            raise Exception("HATA: 'scaled_features' sütunu bulunamadı! Pipeline 1'de hata olabilir.")
            
        # Variance Threshold
        vt = VarianceThresholdSelector(featuresCol="scaled_features", outputCol="vt_features")
        vt.setVarianceThreshold(0.0)
        vt_model = vt.fit(df_windowed)
        df_variance_filtered = vt_model.transform(df_windowed)
        
        # PCA
        pca = PCA(k=PCA_K_DEGERI, inputCol="vt_features", outputCol="features")
        pca_model = pca.fit(df_variance_filtered)
        df_final = pca_model.transform(df_variance_filtered).select("window", "CELL", "N_CELL", "features")
        
        print("Sonuç (İlk 5 Satır):")
        df_final.show(5, truncate=False)
        
        # Varyans Kaydı
        explained_variance = pca_model.explainedVariance.toArray().tolist()
        mlflow.log_param("pca_explained_variance", str(explained_variance))

        # --- 8. GRAFANA VE KAYIT ---
        print("Metrikler Grafana'ya gönderiliyor...")
        g_avg_traffic.set(drift_traffic)
        g_avg_drop.set(drift_drop)
        g_pca_features.set(PCA_K_DEGERI)
        
        try:
            push_to_gateway(PUSHGATEWAY_ADDRESS, job=PROMETHEUS_JOB_NAME, registry=registry)
            print("✓ Grafana metrikleri başarıyla gönderildi.")
        except Exception as e:
            print(f"⚠️ Grafana gönderim hatası: {e}")
            
        # Dosyayı Kaydet
        print(f"Sonuç kaydediliyor: {SON_CIKTI_DOSYASI}")
        df_final.write.mode("overwrite").parquet(SON_CIKTI_DOSYASI)
        mlflow.log_artifact(SON_CIKTI_DOSYASI)
        
        mlflow.log_metric("drift_traffic", drift_traffic)
        mlflow.log_param("is_basarili", "Evet")
        print(f"✓ İŞLEM TAMAMLANDI! Run ID: {run_id}")

    except Exception as e:
        print(f"\n✗ HATA: {e}")
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("error", str(e))
        sys.exit(1)
        
    finally:
        duration = time.time() - start_time
        g_job_duration.set(duration)
        spark.stop()