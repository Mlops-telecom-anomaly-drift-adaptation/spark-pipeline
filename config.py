# TUBITAK Pipeline Configuration
# Bu dosyayı her iki pipeline'a import edebilirsin

import os

# ============================================
# 1. VERİ DOSYA AYARLARI
# ============================================
BUYUK_VERI_SETI_YOLU = "TUBITAK_2807__030825.csv"
OVERLAP_DOSYASI = "Overlap_matrix.csv"
OVERLAP_ESIGI = 40.0  # % cinsinden

# ============================================
# 2. WINDOWING AYARLARI
# ============================================
PENCERE_SURESI = "8 hours"  # "4 hours", "12 hours", "24 hours" deneyin

# ============================================
# 3. FEATURE SELECTION AYARLARI
# ============================================
VARIANCE_THRESHOLD = 0.0  # 0.0 = hiç varyans yok olanaları sil

# PCA bileşen sayısı
# Öneriler:
#   - 50+ feature için: k=10-15
#   - 30-50 feature için: k=5-10
#   - 10-30 feature için: k=3-5
PCA_K_DEGERI = 10

# ============================================
# 4. SPARK AYARLARI
# ============================================
SPARK_APP_NAME_P1 = "DataPrep_Windowing_Full"
SPARK_APP_NAME_P2 = "AdvancedFeatureSelection_Full"

# Spark executor memory (isteğe bağlı)
# SPARK_EXECUTOR_MEMORY = "4g"
# SPARK_DRIVER_MEMORY = "2g"

# ============================================
# 5. MLflow AYARLARI
# ============================================
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_P1 = "Pipeline 1 - Veri Hazirlama (Windowing) - Full Metrics"
MLFLOW_EXPERIMENT_P2 = "Pipeline 2 - Advanced Features (VT & PCA) - Full Metrics"

# MinIO / S3 Ayarları
MLFLOW_S3_ENDPOINT_URL = "http://127.0.0.1:9000"
MLFLOW_S3_ACCESS_KEY = "minioadmin"
MLFLOW_S3_SECRET_KEY = "minioadmin"

# ============================================
# 6. PROMETHEUS / GRAFANA AYARLARI
# ============================================
PROMETHEUS_PUSHGATEWAY = "localhost:9091"
PROMETHEUS_JOB_NAME_P1 = "pipeline_1_data_prep"
PROMETHEUS_JOB_NAME_P2 = "ml_advanced_fs_monitoring_full"

# ============================================
# 7. ÇIKTİ DOSYA İSİMLERİ
# ============================================
OUTPUT_PARQUET_P1 = "windowed_features_output"
OUTPUT_PARQUET_P2 = "ml_ready_data_pca_parquet_full"

# ============================================
# 8. FEATURE SEÇİMİ (Pipeline 2)
# ============================================
# Bu listeyi Pipeline 1'de kullandığın feature adlarına göre update et
SELECTED_FEATURES = [
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

# ============================================
# 9. HATAYOKSULLAŞTİRMA AYARLARI
# ============================================
# True = hataları logla, False = program kapa
CONTINUE_ON_MISSING_FEATURES = True

# ============================================
# 10. DEBUG AYARLARI
# ============================================
VERBOSE_LOGGING = True  # True = detailed logs
SHOW_SAMPLE_DATA = True  # True = örnek verileri print et

# ============================================
# CONFIGURATION YÖNETİMİ
# ============================================

def get_spark_config():
    """Spark session config'i döndür"""
    return {
        "spark.executor.memory": "4g",
        "spark.driver.memory": "2g",
        "spark.sql.shuffle.partitions": "200"
    }

def set_mlflow_env():
    """MLflow environment variable'larını set et"""
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = MLFLOW_S3_ENDPOINT_URL
    os.environ['AWS_ACCESS_KEY_ID'] = MLFLOW_S3_ACCESS_KEY
    os.environ['AWS_SECRET_ACCESS_KEY'] = MLFLOW_S3_SECRET_KEY

def print_config():
    """Mevcut configuration'ı print et"""
    print("\n" + "="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    print(f"Data File: {BUYUK_VERI_SETI_YOLU}")
    print(f"Window Size: {PENCERE_SURESI}")
    print(f"PCA k: {PCA_K_DEGERI}")
    print(f"Feature Count: {len(SELECTED_FEATURES)}")
    print(f"MLflow URI: {MLFLOW_TRACKING_URI}")
    print(f"Prometheus: {PROMETHEUS_PUSHGATEWAY}")
    print("="*60 + "\n")

# ============================================
# ÖRNEK KULLANIM
# ============================================
"""
# pipeline_1.py'de:
from config import *

BUYUK_VERI_SETI_YOLU = BUYUK_VERI_SETI_YOLU
PENCERE_SURESI = PENCERE_SURESI
mlflow.set_experiment(MLFLOW_EXPERIMENT_P1)
print_config()

# pipeline_2.py'de:
from config import *

FEATURE_COLUMNS = SELECTED_FEATURES
PCA_K_DEGERI = PCA_K_DEGERI
mlflow.set_experiment(MLFLOW_EXPERIMENT_P2)
print_config()
"""
