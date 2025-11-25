import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, window
import mlflow

# --- 1. Ayarlar ---
# ŞU ANKİ YENİ VERİ SETİ:
BUYUK_VERI_SETI_YOLU = os.path.join(os.getcwd(), "TUBITAK_2807__030825.csv")
OVERLAP_DOSYASI = os.path.join(os.getcwd(), "Overlap_matrix.csv")
OVERLAP_ESIGI = 40.0
PENCERE_SURESI = "8 hours"
ARA_CIKTI_DOSYASI = "windowed_features_output" 

# --- 2. MLflow & MinIO ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pipeline 1 - Veri Hazirlama (Windowing) - Full Metrics")

# --- 3. Başlat ---
print(f"PIPELINE 1 (Smart Cleaning) başlatılıyor...")
start_time = time.time()
spark = SparkSession.builder.appName("DataPrep_Smart").getOrCreate()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    
    try:
        mlflow.log_param("input_dataset", BUYUK_VERI_SETI_YOLU)
        
        # --- 4. YÜKLE ---
        print(f"Veri yükleniyor: {BUYUK_VERI_SETI_YOLU}")
        df_main = spark.read.csv(BUYUK_VERI_SETI_YOLU, header=True, inferSchema=True, timestampFormat="yyyy-MM-dd HH:mm:ss")
        
        print(f"Overlap verisi yükleniyor: {OVERLAP_DOSYASI}")
        df_overlap_raw = spark.read.csv(OVERLAP_DOSYASI, header=True, inferSchema=True).drop("Unnamed: 0")
        
        # --- 5. AKILLI SÜTUN TEMİZLİĞİ (MAGIC FIX) ---
        print("Sütun isimleri temizleniyor ve standartlaştırılıyor...")
        
        new_column_names = []
        for c in df_main.columns:
            clean_c = c.strip() # Boşlukları sil
            
            # 1. [H] Takısını Sil
            if clean_c.startswith("[H] "):
                clean_c = clean_c.replace("[H] ", "")
            
            # 2. Kritik İsimleri Düzelt
            if clean_c == "SectorID": clean_c = "CELL"
            if clean_c == "SITE": clean_c = "CELL"
            if clean_c == "Traffic Volume Gbyte": clean_c = "Traffic_Volume_Gbyte"
            
            new_column_names.append(clean_c)
        
        # Yeni temiz isimleri DataFrame'e uygula
        df_main = df_main.toDF(*new_column_names)
        
        # Kontrol edelim
        if "CELL" not in df_main.columns:
            raise Exception("HATA: 'CELL' sütunu oluşturulamadı! Dosyadaki hücre kimliği sütunu bulunamadı.")

        # --- 6. İŞLEMLER ---
        print(f"Transformasyon: Overlap Eşiği (>= {OVERLAP_ESIGI}%) uygulanıyor...")
        df_overlap_filtered = df_overlap_raw.filter(col("Overlap_Alan%") >= OVERLAP_ESIGI)
        
        print("Join işlemi yapılıyor...")
        df_joined = df_main.join(df_overlap_filtered.select("CELL", "N_CELL"), on="CELL", how="left")
        
        print("Windowing işlemi yapılıyor...")
        
        # --- AGGREGATION (ARTIK HEPSİ TEMİZ İSİM) ---
        # [H] silindiği için direkt isimleri kullanabiliriz.
        # Noktalı isimler için yine de backtick (`) kullanıyoruz.
        
        df_windowed = df_joined.groupBy(
            window(col("DATETIME"), PENCERE_SURESI),
            col("CELL"),
            col("N_CELL")
        ).agg(
            # Temel Metrikler
            avg("Traffic_Volume_Gbyte").alias("avg_traffic_volume_gbyte_8h"),
            sum("Traffic_Volume_Gbyte").alias("sum_traffic_volume_gbyte_8h"),
            avg("ERAB_Drop_PC").alias("avg_erab_drop_rate_8h"),
            avg("ERAB_ESTAB_ATT").alias("avg_erab_estab_attempts_8h"),
            
            # RRC
            avg("Nof_Avg_SimRRC_ConnUsr").alias("avg_rrc_users_8h"),
            avg("RRC_EstabSucc_PC").alias("avg_rrc_estab_success_rate_8h"),
            avg("NUM_OF_RRC_Att").alias("avg_rrc_attempts_8h"),
            
            # PRB (Varsa)
            avg("DLPRBUtilization").alias("avg_dl_prb_utilization_8h"),
            avg("DL_PRB_Util_%_HWI").alias("avg_dl_prb_util_hwi_8h"),
            
            # Volte (Varsa)
            avg("VOLTE_TRAFFIC_ERL").alias("avg_volte_traffic_erl_8h"),
            
            # İnterferans (Noktalı İsimler - [H] silindiği için böyle çalışır)
            avg(col("`L.UL.Interference.Avg`")).alias("avg_ul_interference_8h"),
            avg(col("`L.UL.Interference.Max`")).alias("max_ul_interference_8h"),
            avg(col("`L.UL.Interference.Min`")).alias("min_ul_interference_8h"),
            
            # Sinyal (Varsa)
            avg("AVGUL_RSSI_WEIGH_DBM_PUCCH").alias("avg_rssi_pucch_8h"),
            avg("AVGUL_RSSI_WEIGH_DBM_PUSCH").alias("avg_rssi_pusch_8h"),
            avg("Avg_UL_RSRP_PUCCH").alias("avg_rsrp_pucch_8h"),
            avg("Avg_UL_RSRP_PUSCH").alias("avg_rsrp_pusch_8h"),
            
            # Diğer
            avg("Avg_CQI_HWI").alias("avg_cqi_8h"),
            avg("HO_Succ_PC_In").alias("avg_ho_success_rate_8h")
        )
        
        df_windowed = df_windowed.na.fill(0)
        
        print(f"Kaydediliyor: {ARA_CIKTI_DOSYASI}")
        df_windowed.write.mode("overwrite").parquet(ARA_CIKTI_DOSYASI)
        
        mlflow.log_artifact(ARA_CIKTI_DOSYASI)
        mlflow.log_param("is_basarili", "Evet")
        mlflow.log_param("total_metrics_count", len(df_windowed.columns))
        
        print(f"MLflow Run ID: {run_id} başarıyla tamamlandı.")
        
        # Sütunları listele
        print("\n=== OLUŞTURULAN ÖZNİTELİKLER ===")
        for i, feat in enumerate(df_windowed.columns, 1):
            print(f"{i}. {feat}")

    except Exception as e:
        print(f"!!!! HATA !!!!")
        print(e)
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("hata_mesaji", str(e))
        sys.exit(1)
        
    finally:
        spark.stop()