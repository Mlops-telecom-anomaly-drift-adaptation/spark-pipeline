import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import VectorAssembler, VarianceThresholdSelector, PCA
import mlflow
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


ARA_GIRDI_DOSYASI = "windowed_features_output"
SON_CIKTI_DOSYASI = "ml_ready_data_pca_parquet_full"


FEATURE_COLUMNS = [
    # --- TRAFİK VE KAPASITITE ---
    "avg_traffic_volume_gbyte_8h",
    "sum_traffic_volume_gbyte_8h",
    
    # --- ERAB METRİKLERİ ---
    "avg_erab_drop_rate_8h",
    "avg_erab_estab_attempts_8h",
    "sum_erab_estab_attempts_8h",
    "avg_erab_estab_success_rate_8h",
    
    # --- RRC METRİKLERİ ---
    "avg_rrc_users_8h",
    "avg_rrc_estab_success_rate_8h",
    "avg_num_rrc_conn_8h",
    "avg_num_rrc_att_8h",
    "avg_rrc_estab_success_8h",
    
    # --- PUSCH METRİKLERİ ---
    "avg_pusch_rrc_count_8h",
    "sum_pusch_rrc_count_8h",
    
    # --- PRB Kullanım Metrikleri ---
    "avg_dl_prb_utilization_8h",
    "avg_ul_prb_utilization_8h",
    "avg_dl_prb_util_8h",
    "avg_dl_prb_percent_8h",
    "avg_prb_rb_used_dl_8h",
    "avg_prb_rb_used_ul_8h",
    
    # --- İnterferans Metrikleri (Downlink) ---
    "avg_ul_interference_8h",
    "max_ul_interference_8h",
    "min_ul_interference_8h",
    
    # --- İnterferans Metrikleri (Uplink - PCC) ---
    "avg_prb0_ul_interference_8h",
    "avg_prb1_ul_interference_8h",
    "avg_prb2_ul_interference_8h",
    
    # --- RSSI (Sinyal Gücü) ---
    "avg_ul_rssi_dbm_8h",
    "avg_ul_rssi_pusch_dbm_8h",
    "avg_ul_rssi_weight_8h",
    
    # --- DBM Metrikleri ---
    "avg_dBm_pucch_8h",
    "avg_dBm_pusch_8h",
    
    # --- Başarı Oranı Metrikleri ---
    "avg_call_success_rate_8h",
    "avg_ho_success_rate_8h",
    "avg_ho_attempts_8h",
    
    # --- Bloke ve Düşen Çağrılar ---
    "avg_blocked_call_pct_8h",
    "avg_dropped_call_pct_8h",
    
    # --- Paket Hatası Metrikleri ---
    "avg_packet_error_rate_8h",
    "avg_mac_dl_ibler_8h",
    "avg_mac_ul_ibler_8h",
    
    # --- Sinyal Kalitesi (RSRP, RSRQ, SINR, CQI) ---
    "avg_rsrp_8h",
    "avg_rsrq_8h",
    "avg_sinr_8h",
    "avg_cqi_8h",
    
    # --- Ek Trafik Metrikleri ---
    "avg_utra_traffic_erl_8h",
    "avg_rach_ta_8h"
]


PCA_K_DEGERI = 10  # Artırıldı - daha fazla variance içermesi için

# --- 2. MLflow & MinIO Sunucu Ayarları ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# MLflow'da ÜÇÜNCÜ bir Deney (Experiment) adı veriyoruz
mlflow.set_experiment("Pipeline 2 - Advanced Features (VT & PCA) - Full Metrics")

# --- 3. Prometheus Pushgateway Ayarları (Grafana) ---
PUSHGATEWAY_ADDRESS = 'localhost:9091'
PROMETHEUS_JOB_NAME = 'ml_advanced_fs_monitoring_full'

registry = CollectorRegistry()
g_job_duration = Gauge('spark_adv_fs_job_duration_seconds', 'Advanced FS İşlem Süresi', registry=registry)
g_job_success = Gauge('spark_adv_fs_job_success', 'Advanced FS Başarı Durumu', registry=registry)
g_avg_traffic = Gauge('spark_adv_fs_avg_traffic_8h', 'Adv FS Ortalama 8 Saatlik Trafik (Data Drift)', registry=registry)
g_feature_count = Gauge('spark_adv_fs_feature_count', 'İşleme giren Feature Sayısı', registry=registry)
g_selected_features_count = Gauge('spark_adv_fs_selected_features_count', 'VT sonrası seçilen Feature Sayısı', registry=registry)
g_pca_features_count = Gauge('spark_adv_fs_pca_features_count', 'PCA sonrası Feature Sayısı', registry=registry)

# --- 4. Spark İşini Başlat ---
print(f"PIPELINE 2 (Advanced Feature Selection - Full Metrics) başlatılıyor...")
start_time = time.time()
spark = SparkSession.builder.appName("AdvancedFeatureSelection_Full").getOrCreate()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    mlflow.log_param("parent_run_id_pipeline_1", "PIPELINE_1_RUN_ID")
    
    try:
        # --- 5. ARA VERİYİ YÜKLE ---
        print(f"Pipeline 1'den gelen ara dosya okunuyor: {ARA_GIRDI_DOSYASI}")
        try:
            df_windowed = spark.read.parquet(ARA_GIRDI_DOSYASI)
        except Exception as e:
            print(f"HATA: '{ARA_GIRDI_DOSYASI}' okunamadı. Önce pipeline_1_data_prep_full.py'yi çalıştırdınız mı?")
            raise e
        
        print(f"Veri başarıyla yüklendi.")
        print(f"Sütun sayısı: {len(df_windowed.columns)}")
        
        # Mevcut olan özellikleri kontrol et
        available_columns = df_windowed.columns
        valid_features = [f for f in FEATURE_COLUMNS if f in available_columns]
        missing_features = [f for f in FEATURE_COLUMNS if f not in available_columns]
        
        if missing_features:
            print(f"\nUyarı: Aşağıdaki öznitelikler bulunamadı (Pipeline 1'de hesaplanmamış olabilir):")
            for mf in missing_features:
                print(f"  - {mf}")
            print(f"\nDevam edilecek öznitelikler ({len(valid_features)} adet):")
        
        FEATURE_COLUMNS = valid_features
        
        for i, feat in enumerate(FEATURE_COLUMNS, 1):
            print(f"{i}. {feat}")
        
        # --- 6. DATA DRIFT METRİKLERİNİ HESAPLA ---
        print("\n" + "="*60)
        print("Data Drift/Grafana metrikleri hesaplanıyor...")
        df_drift_metrics = df_windowed.agg(avg("avg_traffic_volume_gbyte_8h").alias("drift_avg_traffic")).first()
        drift_traffic = df_drift_metrics["drift_avg_traffic"] if df_drift_metrics["drift_avg_traffic"] else 0
        
        # --- 7. YENİ TRANSFORMASYONLAR (VectorAssembler -> VT -> PCA) ---
        print("\n" + "="*60)
        print("Transformasyonlar başlıyor...")
        
        # Adım 7a: Tüm öznitelikleri bir vektörde topla
        print("\nAdım 7a: VectorAssembler çalışıyor...")
        assembler = VectorAssembler(inputCols=FEATURE_COLUMNS, outputCol="temp_features")
        df_assembled = assembler.transform(df_windowed)
        print(f"✓ VectorAssembler başarılı - {len(FEATURE_COLUMNS)} öznitelik birleştirildi")
        
        # Adım 7b: VarianceThresholdSelector
        print("\nAdım 7b: VarianceThresholdSelector çalışıyor...")
        vt = VarianceThresholdSelector(featuresCol="temp_features", outputCol="vt_features")
        vt.setVarianceThreshold(0.0)
        vt_model = vt.fit(df_assembled)
        df_variance_filtered = vt_model.transform(df_assembled)
        
        selected_features_indices = vt_model.selectedFeatures
        selected_features = [FEATURE_COLUMNS[i] for i in selected_features_indices]
        print(f"✓ Varyans Eşiği uygulandı")
        print(f"  - Başlangıç feature sayısı: {len(FEATURE_COLUMNS)}")
        print(f"  - Seçilen feature sayısı: {len(selected_features)}")
        print(f"  - Silinen feature sayısı: {len(FEATURE_COLUMNS) - len(selected_features)}")
        
        if len(selected_features) <= 15:
            print(f"\n  Seçilen öznitelikler:")
            for i, feat in enumerate(selected_features, 1):
                print(f"    {i}. {feat}")
        
        mlflow.log_param("vt_selected_features", str(selected_features))
        mlflow.log_param("vt_features_count", len(selected_features))
        mlflow.log_param("vt_removed_features_count", len(FEATURE_COLUMNS) - len(selected_features))
        
        # Adım 7c: PCA
        print(f"\nAdım 7c: PCA (k={PCA_K_DEGERI}) çalışıyor...")
        
        # Seçilen feature sayısı PCA_K_DEGERI'nden azsa, onu adjust et
        actual_pca_k = min(PCA_K_DEGERI, len(selected_features))
        
        pca = PCA(k=actual_pca_k, inputCol="vt_features", outputCol="features")
        pca_model = pca.fit(df_variance_filtered)
        df_final = pca_model.transform(df_variance_filtered).select("window", "CELL", "N_CELL", "features")
        
        print(f"✓ PCA başarıyla uygulandı (k={actual_pca_k})")
        
        # PCA açıklanan varyansı
        explained_variance = pca_model.explainedVariance.toArray().tolist()
        total_variance = sum(explained_variance)
        print(f"\n  PCA Açıklanan Varyans (Component-wise):")
        for i, var in enumerate(explained_variance, 1):
            print(f"    PC-{i}: {var:.4f} ({var/total_variance*100:.2f}%)")
        print(f"  Toplam Açıklanan Varyans: {total_variance:.4f} ({total_variance/len(explained_variance)*100:.2f}%)")
        
        print("\nML'e hazır son (PCA uygulanmış) veri (ilk 5 satır):")
        df_final.show(5, truncate=False)
        
        mlflow.log_param("pca_k_degeri", actual_pca_k)
        mlflow.log_param("pca_explained_variance", str(explained_variance))
        mlflow.log_param("pca_total_variance_explained", f"{total_variance:.4f}")
        
        # --- 8. METRİKLERİ KAYDET (MLflow & Prometheus) ---
        print("\n" + "="*60)
        print("Metrikler MLflow'a ve Prometheus'a gönderiliyor...")
        
        output_row_count = df_final.count()
        
        mlflow.log_metric("input_feature_count", len(FEATURE_COLUMNS))
        mlflow.log_metric("vt_selected_features_count", len(selected_features))
        mlflow.log_metric("pca_components_count", actual_pca_k)
        mlflow.log_metric("drift_avg_traffic", drift_traffic)
        mlflow.log_metric("output_row_count", output_row_count)
        
        # Prometheus metrikleri
        g_feature_count.set(len(FEATURE_COLUMNS))
        g_selected_features_count.set(len(selected_features))
        g_pca_features_count.set(actual_pca_k)
        g_avg_traffic.set(drift_traffic)
        g_job_success.set(1)
        
        print(f"✓ Metrikler kaydedildi")
        print(f"  - Input Features: {len(FEATURE_COLUMNS)}")
        print(f"  - VT Sonrası: {len(selected_features)}")
        print(f"  - PCA Sonrası: {actual_pca_k}")
        print(f"  - Çıktı Satır Sayısı: {output_row_count}")
        
        # --- 9. SON ARTIFACT'İ KAYDET (MLflow/MinIO) ---
        print("\n" + "="*60)
        print(f"ML'e hazır son veri ({SON_CIKTI_DOSYASI}) MinIO'ya yüklenmek üzere kaydediliyor...")
        
        df_final.write.mode("overwrite").parquet(SON_CIKTI_DOSYASI)
        mlflow.log_artifact(SON_CIKTI_DOSYASI)
        
        mlflow.log_param("is_basarili", "Evet")
        print(f"\n✓ MLflow Run ID: {run_id} başarıyla tamamlandı (Pipeline 2 - Advanced - Full Metrics).")
        
    except Exception as e:
        print(f"\n!!!! PIPELINE 2'DE HATA OLUŞTU !!!!")
        print(e)
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("hata_mesaji", str(e))
        g_job_success.set(0)
        sys.exit(1)
        
    finally:
        duration = time.time() - start_time
        print(f"\nİş (Pipeline 2 - Advanced) {duration:.2f} saniye sürdü.")
        mlflow.log_metric("job_duration_seconds", duration)
        
        try:
            push_to_gateway(PUSHGATEWAY_ADDRESS, job=PROMETHEUS_JOB_NAME, registry=registry)
            print("✓ Prometheus metrikleri başarıyla gönderildi.")
        except Exception as e:
            print(f"⚠ Prometheus Pushgateway'e gönderilemedi (Pushgateway açık mı?): {e}")
        
        print("Spark Session durduruluyor (Pipeline 2 - Advanced).")
        spark.stop()
