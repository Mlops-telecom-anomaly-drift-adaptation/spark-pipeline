import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
# YENİ: RobustScaler eklendi
from pyspark.ml.feature import VectorAssembler, VarianceThresholdSelector, PCA, RobustScaler
import mlflow
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# --- 1. Ayarlar ---
ARA_GIRDI_DOSYASI = "windowed_features_output" 
SON_CIKTI_DOSYASI = "ml_ready_data_pca_parquet_full" 

# Pipeline 1'den gelen GERÇEK sütun isimleri (Uzun isimler)
FEATURE_COLUMNS = [
    # Trafik
    "avg_traffic_volume_gbyte_8h",
    "sum_traffic_volume_gbyte_8h",
    
    # Bağlantı Kalitesi
    "avg_erab_drop_rate_8h",
    "avg_erab_estab_attempts_8h",
    "avg_erab_setup_success_rate_8h",
    
    # RRC / Kullanıcı
    "avg_rrc_users_8h",
    "avg_rrc_estab_success_rate_8h",
    "avg_rrc_attempts_8h",
    "avg_sim_rrc_conn_count_8h",
    
    # Kaynak Kullanımı
    "avg_dl_prb_utilization_8h",
    "avg_ul_prb_utilization_8h",
    "avg_dl_prb_util_hwi_8h",
    
    # İnterferans / Gürültü
    "avg_ul_interference_8h",
    "max_ul_interference_8h",
    "min_ul_interference_8h",
    
    # Sinyal Gücü (RSSI)
    "avg_rssi_pucch_8h",
    "avg_rssi_pusch_8h",
    
    # Sinyal Kalitesi
    "avg_rsrp_pusch_8h",
    "avg_rsrp_pucch_8h",
    "avg_cqi_8h",
    
    # Diğerleri
    "avg_volte_traffic_erl_8h",
    "avg_ho_success_in_8h",
    "avg_intrafreq_ho_sr_8h",
    "avg_interfreq_ho_sr_8h",
    "avg_ul_packet_loss_8h"
]

PCA_K_DEGERI = 2

# --- 2. MLflow & MinIO Sunucu Ayarları ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pipeline 2 - Advanced Features (VT & PCA) - Full Metrics")

# --- 3. Prometheus Pushgateway Ayarları (Grafana) ---
PUSHGATEWAY_ADDRESS = 'localhost:9091'
PROMETHEUS_JOB_NAME = 'ml_advanced_fs_monitoring_full' 
registry = CollectorRegistry()

# Grafana Metrikleri
g_job_duration = Gauge('spark_adv_fs_job_duration_seconds', 'Advanced FS Islem Suresi', registry=registry)
g_job_success = Gauge('spark_adv_fs_job_success', 'Advanced FS Basari Durumu', registry=registry)
g_avg_traffic = Gauge('spark_adv_fs_avg_traffic_8h', 'Ortalama 8 Saatlik Trafik (Data Drift)', registry=registry)
g_avg_drop = Gauge('spark_adv_fs_avg_erab_drop_rate_8h', 'Ortalama Dusme Orani (Data Drift)', registry=registry)
g_feature_count = Gauge('spark_adv_fs_feature_count', 'Girdi Feature Sayisi', registry=registry)
g_pca_features = Gauge('spark_adv_fs_pca_features_count', 'PCA Sonrasi Feature Sayisi', registry=registry)

# --- 4. Spark İşini Başlat ---
print(f"PIPELINE 2 (Advanced Feature Selection - Full Metrics) başlatılıyor...")
start_time = time.time()
spark = SparkSession.builder.appName("AdvancedFeatureSelection_Full").getOrCreate()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    
    try:
        # --- 5. ARA VERİYİ YÜKLE ---
        print(f"Pipeline 1'den gelen ara dosya okunuyor: {ARA_GIRDI_DOSYASI}")
        try:
            df_windowed = spark.read.parquet(ARA_GIRDI_DOSYASI)
        except Exception as e:
            print(f"HATA: Ara dosya okunamadı. Lütfen önce Pipeline 1'i çalıştırın.")
            raise e

        # Sütun kontrolü
        available_cols = df_windowed.columns
        valid_features = [c for c in FEATURE_COLUMNS if c in available_cols]
        
        print(f"Mevcut Sütun Sayısı: {len(available_cols)}")
        print(f"Kullanılacak Feature Sayısı: {len(valid_features)}")
        
        if len(valid_features) == 0:
             raise Exception("HİÇBİR GEÇERLİ FEATURE BULUNAMADI! Pipeline 1 çıktısını kontrol edin.")
        
        # --- 6. DATA DRIFT METRİKLERİNİ HESAPLA ---
        print("Data Drift metrikleri hesaplanıyor...")
        
        # Drift için kullanılacak sütunları kontrol et (varsa kullan)
        drift_traffic = 0
        drift_drop = 0
        
        if "avg_traffic_volume_gbyte_8h" in available_cols:
            drift_traffic = df_windowed.agg(avg("avg_traffic_volume_gbyte_8h")).first()[0] or 0
            
        if "avg_erab_drop_rate_8h" in available_cols:
            drift_drop = df_windowed.agg(avg("avg_erab_drop_rate_8h")).first()[0] or 0

        # --- 7. TRANSFORMASYONLAR ---
        print("Adım 7a: VectorAssembler çalışıyor...")
        assembler = VectorAssembler(inputCols=valid_features, outputCol="raw_features")
        df_assembled = assembler.transform(df_windowed)

        # YENİ ADIM: RobustScaler (Outlier Koruması)
        print("Adım 7b: RobustScaler (Aykırı Değer Koruması) çalışıyor...")
        # withScaling=True (IQR ile ölçekle), withCentering=True (Medyanı çıkar)
        scaler = RobustScaler(inputCol="raw_features", outputCol="scaled_features", 
                              withScaling=True, withCentering=True, 
                              lower=0.25, upper=0.75)
        scaler_model = scaler.fit(df_assembled)
        df_scaled = scaler_model.transform(df_assembled)

        print("Adım 7c: VarianceThresholdSelector çalışıyor...")
        # Scaler'dan çıkan 'scaled_features' sütununu kullanıyoruz
        vt = VarianceThresholdSelector(featuresCol="scaled_features", outputCol="vt_features")
        vt.setVarianceThreshold(0.0) # Sıfır varyanslıları at
        vt_model = vt.fit(df_scaled)
        df_variance_filtered = vt_model.transform(df_scaled)
        
        print("Adım 7d: PCA çalışıyor...")
        actual_k = min(PCA_K_DEGERI, len(valid_features))
        pca = PCA(k=actual_k, inputCol="vt_features", outputCol="features")
        pca_model = pca.fit(df_variance_filtered)
        df_final = pca_model.transform(df_variance_filtered).select("window", "CELL", "N_CELL", "features")
        
        print("ML'e hazır son veri (ilk 5 satır):")
        df_final.show(5, truncate=False)
        
        # Varyans oranlarını kaydet
        explained_variance = pca_model.explainedVariance.toArray().tolist()
        print(f"PCA Açıklanan Varyans: {explained_variance}")
        mlflow.log_param("pca_explained_variance", str(explained_variance))
        mlflow.log_param("scaler_type", "RobustScaler")

        # --- 8. METRİKLERİ KAYDET ---
        print("Metrikler Grafana'ya gönderiliyor...")
        g_avg_traffic.set(drift_traffic)
        g_avg_drop.set(drift_drop)
        g_feature_count.set(len(valid_features))
        g_pca_features.set(actual_k)
        g_job_success.set(1)
        
        mlflow.log_metric("drift_avg_traffic", drift_traffic)
        mlflow.log_metric("drift_avg_drop_rate", drift_drop)

        # --- 9. SON ARTIFACT'İ KAYDET ---
        print(f"Sonuç kaydediliyor: {SON_CIKTI_DOSYASI}")
        df_final.write.mode("overwrite").parquet(SON_CIKTI_DOSYASI)
        mlflow.log_artifact(SON_CIKTI_DOSYASI)

        mlflow.log_param("is_basarili", "Evet")
        print(f"MLflow Run ID: {run_id} başarıyla tamamlandı.")

    except Exception as e:
        print(f"!!!! HATA OLUŞTU !!!!")
        print(e)
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("hata_mesaji", str(e))
        g_job_success.set(0)
        sys.exit(1)

    finally:
        duration = time.time() - start_time
        mlflow.log_metric("job_duration_seconds", duration)
        g_job_duration.set(duration)
        try:
            push_to_gateway(PUSHGATEWAY_ADDRESS, job=PROMETHEUS_JOB_NAME, registry=registry)
            print("Grafana metrikleri gönderildi.")
        except:
            print("Grafana'ya gönderilemedi.")
            
        spark.stop()