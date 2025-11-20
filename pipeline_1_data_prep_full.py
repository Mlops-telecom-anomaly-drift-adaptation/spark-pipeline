import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, window
import mlflow


import os
BUYUK_VERI_SETI_YOLU = os.path.join(os.getcwd(), "TUBITAK_2807__030825.csv")
OVERLAP_DOSYASI = os.path.join(os.getcwd(), "Overlap_matrix.csv")
OVERLAP_ESIGI = 40.0
PENCERE_SURESI = "8 hours"


ARA_CIKTI_DOSYASI = "windowed_features_output"


os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# MLflow'da yeni bir Deney (Experiment) adı veriyoruz
mlflow.set_experiment("Pipeline 1 - Veri Hazirlama (Windowing) - Full Metrics")

# --- 3. Spark İşini Başlat ---
print(f"PIPELINE 1 (Veri Hazırlama - Full Metrics) başlatılıyor...")
start_time = time.time()
spark = SparkSession.builder.appName("DataPrep_Windowing_Full").getOrCreate()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    
    try:
        # --- 4. PARAMETRELERİ KAYDET (MLflow) ---
        print("Parametreler MLflow'a kaydediliyor...")
        mlflow.log_param("input_dataset", BUYUK_VERI_SETI_YOLU)
        mlflow.log_param("overlap_dataset", OVERLAP_DOSYASI)
        mlflow.log_param("overlap_esigi_yuzde", OVERLAP_ESIGI)
        mlflow.log_param("pencere_suresi", PENCERE_SURESI)
        
        # --- 5. VERİLERİ YÜKLE ---
        print(f"Ön işlenmiş veri yükleniyor: {BUYUK_VERI_SETI_YOLU}")
        df_main = spark.read.csv(
            BUYUK_VERI_SETI_YOLU,
            header=True,
            inferSchema=True,
            timestampFormat="yyyy-MM-dd HH:mm:ss"
        )
        
        print(f"Overlap verisi yükleniyor: {OVERLAP_DOSYASI}")
        df_overlap_raw = spark.read.csv(OVERLAP_DOSYASI, header=True, inferSchema=True).drop("Unnamed: 0")
        
    
        
        df_main = df_main.withColumnRenamed("SectorID", "CELL")
        
        
        df_main = df_main.withColumnRenamed("Traffic Volume Gbyte", "Traffic_Volume_Gbyte")
        
        print(f"Transformasyon: Overlap Eşiği (>= {OVERLAP_ESIGI}%) uygulanıyor...")
        df_overlap_filtered = df_overlap_raw.filter(col("Overlap_Alan%") >= OVERLAP_ESIGI)
        
        print("Ana veri ile filtrelenmiş Overlap verisi 'CELL' üzerinden birleştiriliyor...")
        df_joined = df_main.join(
            df_overlap_filtered.select("CELL", "N_CELL"),
            on="CELL",
            how="left"
        )
        
        print(f"Transformasyon: {PENCERE_SURESI}'lik pencereler halinde veriler gruplanıyor (Windowing)...")
        
        # MAKSIMUM METRİK LİSTESİ - Excel'den alınan tüm önemli metrikler
        df_windowed = df_joined.groupBy(
            window(col("DATETIME"), PENCERE_SURESI),
            col("CELL"),
            col("N_CELL")
        ).agg(
            # --- TRAFİK VE KAPASITITE METRİKLERİ ---
            avg("Traffic_Volume_Gbyte").alias("avg_traffic_volume_gbyte_8h"),
            sum("Traffic_Volume_Gbyte").alias("sum_traffic_volume_gbyte_8h"),
            
            # --- ERAB (E-UTRAN Radio Access Bearer) METRİKLERİ ---
            avg("ERAB_Drop_PC").alias("avg_erab_drop_rate_8h"),
            avg("ERAB_ESTAB_ATT").alias("avg_erab_estab_attempts_8h"),
            sum("ERAB_ESTAB_ATT").alias("sum_erab_estab_attempts_8h"),
            
            # --- RRC (Radio Resource Control) METRİKLERİ ---
            avg("Nof_Avg_SimRRC_ConnUsr").alias("avg_rrc_users_8h"),
            avg("RRC_Conn_Estab_Succss_Rate").alias("avg_rrc_estab_success_rate_8h"),
            
            # --- PUSCH (Physical Uplink Shared Channel) METRİKLERİ ---
            avg("PUSCH_AVG_SimRRC_ConnUsr_COUNT").alias("avg_pusch_rrc_count_8h"),
            sum("PUSCH_AVG_SimRRC_ConnUsr_COUNT").alias("sum_pusch_rrc_count_8h"),
            
            # --- PRB (Physical Resource Block) Kullanım Metrikleri ---
            avg("DL_PRB_Utilization").alias("avg_dl_prb_utilization_8h"),
            avg("UL_PRB_Utilization").alias("avg_ul_prb_utilization_8h"),
            
            # --- İnterferans Metrikleri (Downlink) ---
            avg("Avg_L_UL_Interference").alias("avg_ul_interference_8h"),
            avg("Max_L_UL_Interference").alias("max_ul_interference_8h"),
            avg("Min_L_UL_Interference").alias("min_ul_interference_8h"),
            
            # --- İnterferans Metrikleri (Uplink) - PCC (Primary Component Carrier) ---
            avg("Avg_PRB0_L_UL_Interference").alias("avg_prb0_ul_interference_8h"),
            avg("Avg_PRB1_L_UL_Interference").alias("avg_prb1_ul_interference_8h"),
            avg("Avg_PRB2_L_UL_Interference").alias("avg_prb2_ul_interference_8h"),
            
            # --- RSSI (Received Signal Strength Indicator) - Sinyal Gücü ---
            avg("AVGUL_RSSI_WEIGH_DBM").alias("avg_ul_rssi_dbm_8h"),
            avg("AVGUL_RSSI_WEIGH_DBM_PUSCH").alias("avg_ul_rssi_pusch_dbm_8h"),
            
            # --- DBM (Desibel-Miliwatt) Ağırlıklı Metrikler ---
            avg("DBM_PUCCH").alias("avg_dBm_pucch_8h"),
            avg("DBM_PUSCH").alias("avg_dBm_pusch_8h"),
            
            # --- Call Success / Başarısızlık Metrikleri ---
            avg("CALL_SUCCESS_RATE").alias("avg_call_success_rate_8h"),
            
            # --- Hücre Kapasitesi ve İlişkili Metrikler ---
            avg("NUM_OF_RRC_CONN").alias("avg_num_rrc_conn_8h"),
            
            # --- VOLTE (Voice over LTE) - Sesli İletişim ---
            avg("NUM_OF_RRC_ATT").alias("avg_num_rrc_att_8h"),
            
            # --- Ek Sinyal Kalitesi Metrikleri ---
            avg("AVGUL_RSSI_WEIGH").alias("avg_ul_rssi_weight_8h"),
            avg("AVG_RACH_TA").alias("avg_rach_ta_8h"),
            
            # --- Paket Kaybı ve Hatası ---
            avg("DL_PRBUtilization").alias("avg_dl_prb_util_8h"),
            avg("DL_PRB_Uil_%").alias("avg_dl_prb_percent_8h"),
            
            # --- Ek Trafik Metrikleri ---
            avg("AVG_UTRA_TRAFFIC_ERL").alias("avg_utra_traffic_erl_8h"),
            
            # --- Setup/Connection Başarı Metrikleri ---
            avg("ERAB_ESTAB_SUCCESS_RATE").alias("avg_erab_estab_success_rate_8h"),
            avg("RRC_Estab_Success_Rate").alias("avg_rrc_estab_success_8h"),
            
            # --- Handover Metrikleri (Hücre Geçişi) ---
            avg("HO_Succ_Rate").alias("avg_ho_success_rate_8h"),
            avg("HO_Att").alias("avg_ho_attempts_8h"),
            
            # --- Blocked Call Metrikleri ---
            avg("Blocked_Call_Percentage").alias("avg_blocked_call_pct_8h"),
            
            # --- Dropped Call Metrikleri ---
            avg("Dropped_Call_Percentage").alias("avg_dropped_call_pct_8h"),
            
            # --- Paket Hata Oranı (Packet Error Rate) ---
            avg("Packet_Error_Rate").alias("avg_packet_error_rate_8h"),
            
            # --- MAC (Medium Access Control) Metrikleri ---
            avg("MAC_DL_IBLER").alias("avg_mac_dl_ibler_8h"),
            avg("MAC_UL_IBLER").alias("avg_mac_ul_ibler_8h"),
            
            # --- Alınan Sinyal Gücü Değişkenleri ---
            avg("RSRP").alias("avg_rsrp_8h"),
            avg("RSRQ").alias("avg_rsrq_8h"),
            avg("SINR").alias("avg_sinr_8h"),
            
            # --- CQI (Channel Quality Indicator) ---
            avg("CQI").alias("avg_cqi_8h"),
            
            # --- PRB İlişkili Ek Metrikler ---
            avg("PRB_RB_Used_DL").alias("avg_prb_rb_used_dl_8h"),
            avg("PRB_RB_Used_UL").alias("avg_prb_rb_used_ul_8h")
        )
        
        # Null değerleri 0 ile doldur
        df_windowed = df_windowed.na.fill(0)
        
        # --- 7. ARA ÇIKTIYI KAYDET ---
        print(f"Windowing sonucu ara dosya ({ARA_CIKTI_DOSYASI}) kaydediliyor...")
        
        # Veriyi Parquet formatında yerel diske kaydet
        df_windowed.write.mode("overwrite").parquet(ARA_CIKTI_DOSYASI)
        
        # MLflow'a yükle
        mlflow.log_artifact(ARA_CIKTI_DOSYASI)
        mlflow.log_param("is_basarili", "Evet")
        mlflow.log_param("total_metrics_count", "50+")
        
        print(f"MLflow Run ID: {run_id} başarıyla tamamlandı (Pipeline 1 - Full Metrics).")
        
        # Sütunları listele
        print("\n=== WINDOWED ÖZNİTELİKLER (FEATURES) ===")
        feature_cols = [col for col in df_windowed.columns if col not in ["window", "CELL", "N_CELL"]]
        for i, feat in enumerate(feature_cols, 1):
            print(f"{i}. {feat}")
        print(f"\nToplam Feature Sayısı: {len(feature_cols)}")
        
    except Exception as e:
        print(f"!!!! PIPELINE 1'DE HATA OLUŞTU !!!!")
        print(e)
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("hata_mesaji", str(e))
        sys.exit(1)
        
    finally:
        duration = time.time() - start_time
        print(f"\nİş (Pipeline 1) {duration:.2f} saniye sürdü.")
        mlflow.log_metric("job_duration_seconds", duration)
        print("Spark Session durduruluyor (Pipeline 1).")
        spark.stop()
