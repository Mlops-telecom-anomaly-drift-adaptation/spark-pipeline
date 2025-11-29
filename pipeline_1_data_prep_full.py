import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, window
from sklearn.preprocessing import RobustScaler
import pandas as pd
import mlflow

# --- 1. Ayarlar ---
BUYUK_VERI_SETI_YOLU = os.path.join(os.getcwd(), "TUBITAK_2807__030825.csv")
OVERLAP_DOSYASI = os.path.join(os.getcwd(), "Overlap_matrix.csv")
OVERLAP_ESIGI = 40.0
PENCERE_SURESI = "8 hours"
ARA_CIKTI_DOSYASI = "windowed_features_output" 
ROBUST_SCALED_OUTPUT = "pipeline1_robust_scaled_data"

# --- 2. MLflow & MinIO ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pipeline 1 - Veri Hazirlama (RobustScaler) - Full Metrics")

# --- 3. BaÅŸlat ---
print(f"PIPELINE 1 (RobustScaler) baÅŸlatÄ±lÄ±yor...")
start_time = time.time()
spark = SparkSession.builder.appName("DataPrep_RobustScaler").getOrCreate()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} baÅŸladÄ±.")
    
    try:
        mlflow.log_param("input_dataset", BUYUK_VERI_SETI_YOLU)
        mlflow.log_param("scaler_type", "RobustScaler")
        
        # --- 4. YÃœKLE ---
        print(f"Veri yÃ¼kleniyor: {BUYUK_VERI_SETI_YOLU}")
        df_main = spark.read.csv(BUYUK_VERI_SETI_YOLU, header=True, inferSchema=True, timestampFormat="yyyy-MM-dd HH:mm:ss")
        
        print(f"Overlap verisi yÃ¼kleniyor: {OVERLAP_DOSYASI}")
        df_overlap_raw = spark.read.csv(OVERLAP_DOSYASI, header=True, inferSchema=True).drop("Unnamed: 0")
        
        # --- 5. AKILLI SÃœTUN TEMÄ°ZLÄ°ÄžÄ° ---
        print("SÃ¼tun isimleri temizleniyor ve standartlaÅŸtÄ±rÄ±lÄ±yor...")
        
        new_column_names = []
        for c in df_main.columns:
            clean_c = c.strip()
            
            if clean_c.startswith("[H] "):
                clean_c = clean_c.replace("[H] ", "")
            
            if clean_c == "SectorID": clean_c = "CELL"
            if clean_c == "SITE": clean_c = "CELL"
            if clean_c == "Traffic Volume Gbyte": clean_c = "Traffic_Volume_Gbyte"
            
            new_column_names.append(clean_c)
        
        df_main = df_main.toDF(*new_column_names)
        
        if "CELL" not in df_main.columns:
            raise Exception("HATA: 'CELL' sÃ¼tunu oluÅŸturulamadÄ±!")

        # --- 6. Ä°ÅžLEMLER ---
        print(f"Transformasyon: Overlap EÅŸiÄŸi (>= {OVERLAP_ESIGI}%) uygulanÄ±yor...")
        df_overlap_filtered = df_overlap_raw.filter(col("Overlap_Alan%") >= OVERLAP_ESIGI)
        
        print("Join iÅŸlemi yapÄ±lÄ±yor...")
        df_joined = df_main.join(df_overlap_filtered.select("CELL", "N_CELL"), on="CELL", how="left")
        
        print("Windowing iÅŸlemi yapÄ±lÄ±yor...")
        
        df_windowed = df_joined.groupBy(
            window(col("DATETIME"), PENCERE_SURESI),
            col("CELL"),
            col("N_CELL")
        ).agg(
            avg("Traffic_Volume_Gbyte").alias("avg_traffic_volume_gbyte_8h"),
            sum("Traffic_Volume_Gbyte").alias("sum_traffic_volume_gbyte_8h"),
            avg("ERAB_Drop_PC").alias("avg_erab_drop_rate_8h"),
            avg("ERAB_ESTAB_ATT").alias("avg_erab_estab_attempts_8h"),
            avg("Nof_Avg_SimRRC_ConnUsr").alias("avg_rrc_users_8h"),
            avg("RRC_EstabSucc_PC").alias("avg_rrc_estab_success_rate_8h"),
            avg("NUM_OF_RRC_Att").alias("avg_rrc_attempts_8h"),
            avg("DLPRBUtilization").alias("avg_dl_prb_utilization_8h"),
            avg("DL_PRB_Util_%_HWI").alias("avg_dl_prb_util_hwi_8h"),
            avg("VOLTE_TRAFFIC_ERL").alias("avg_volte_traffic_erl_8h"),
            avg(col("`L.UL.Interference.Avg`")).alias("avg_ul_interference_8h"),
            avg(col("`L.UL.Interference.Max`")).alias("max_ul_interference_8h"),
            avg(col("`L.UL.Interference.Min`")).alias("min_ul_interference_8h"),
            avg("AVGUL_RSSI_WEIGH_DBM_PUCCH").alias("avg_rssi_pucch_8h"),
            avg("AVGUL_RSSI_WEIGH_DBM_PUSCH").alias("avg_rssi_pusch_8h"),
            avg("Avg_UL_RSRP_PUCCH").alias("avg_rsrp_pucch_8h"),
            avg("Avg_UL_RSRP_PUSCH").alias("avg_rsrp_pusch_8h"),
            avg("Avg_CQI_HWI").alias("avg_cqi_8h"),
            avg("HO_Succ_PC_In").alias("avg_ho_success_rate_8h")
        )
        
        df_windowed = df_windowed.na.fill(0)
        
        # --- 7. ROBUST SCALER UYGULAMASI ---
        print("\n=== ROBUST SCALER UYGULANIYYOR ===")
        
        # Spark DF'i Pandas'a dönüştür
        df_pandas = df_windowed.toPandas()
        print(f"Toplam satır: {len(df_pandas)}")
        print(f"Toplam sütun: {len(df_pandas.columns)}")
        
        # Numeral sütunları belirle (window, CELL, N_CELL hariç)
        numeric_cols = df_pandas.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # window, CELL, N_CELL hariç tut
        numeric_cols = [col for col in numeric_cols if col not in ['CELL', 'N_CELL']]
        
        print(f"\nÖlçeklenen sütunlar: {len(numeric_cols)}")
        
        # RobustScaler öncesi istatistikler
        print("\n--- ÖNCESİ (Original Data) ---")
        original_mean = df_pandas[numeric_cols].mean().mean()
        original_std = df_pandas[numeric_cols].std().mean()
        original_median = df_pandas[numeric_cols].median().mean()
        original_q1 = df_pandas[numeric_cols].quantile(0.25).mean()
        original_q3 = df_pandas[numeric_cols].quantile(0.75).mean()
        original_iqr = original_q3 - original_q1
        
        print(f"Mean: {original_mean:.6f}")
        print(f"Std: {original_std:.6f}")
        print(f"Median: {original_median:.6f}")
        print(f"Q1 (25%): {original_q1:.6f}")
        print(f"Q3 (75%): {original_q3:.6f}")
        print(f"IQR: {original_iqr:.6f}")
        
        # RobustScaler uygula
        scaler = RobustScaler()
        df_scaled = df_pandas.copy()
        df_scaled[numeric_cols] = scaler.fit_transform(df_pandas[numeric_cols])
        
        # RobustScaler sonrası istatistikler
        print("\n--- SONRASI (After RobustScaler) ---")
        scaled_mean = df_scaled[numeric_cols].mean().mean()
        scaled_std = df_scaled[numeric_cols].std().mean()
        scaled_median = df_scaled[numeric_cols].median().mean()
        scaled_q1 = df_scaled[numeric_cols].quantile(0.25).mean()
        scaled_q3 = df_scaled[numeric_cols].quantile(0.75).mean()
        scaled_iqr = scaled_q3 - scaled_q1
        
        print(f"Mean: {scaled_mean:.6f}")
        print(f"Std: {scaled_std:.6f}")
        print(f"Median: {scaled_median:.6f}")
        print(f"Q1 (25%): {scaled_q1:.6f}")
        print(f"Q3 (75%): {scaled_q3:.6f}")
        print(f"IQR: {scaled_iqr:.6f}")
        
        # --- 8. KAYDET ---
        print(f"\nKaydediliyor: {ARA_CIKTI_DOSYASI} (Orijinal windowed)")
        spark.createDataFrame(df_pandas).write.mode("overwrite").parquet(ARA_CIKTI_DOSYASI)
        
        print(f"Kaydediliyor: {ROBUST_SCALED_OUTPUT} (RobustScaler uygulanmış)")
        df_scaled.to_parquet(f"{ROBUST_SCALED_OUTPUT}/scaled_data.parquet")
        df_scaled.to_csv(f"{ROBUST_SCALED_OUTPUT}_data.csv", index=False)
        
        # --- 9. MLFLOW LOGGING ---
        print("\nMLflow'a metrikler kaydediliyor...")
        
        # Orijinal veri metrikleri
        mlflow.log_metric("original_mean", original_mean)
        mlflow.log_metric("original_std", original_std)
        mlflow.log_metric("original_median", original_median)
        mlflow.log_metric("original_iqr", original_iqr)
        
        # Scaled veri metrikleri
        mlflow.log_metric("scaled_mean", scaled_mean)
        mlflow.log_metric("scaled_std", scaled_std)
        mlflow.log_metric("scaled_median", scaled_median)
        mlflow.log_metric("scaled_iqr", scaled_iqr)
        
        # RobustScaler center ve scale parametreleri
        mlflow.log_param("scaler_center", str(scaler.center_[:5]))  # İlk 5
        mlflow.log_param("scaler_scale", str(scaler.scale_[:5]))    # İlk 5
        
        # Genel parametreler
        mlflow.log_param("total_samples", len(df_scaled))
        mlflow.log_param("total_numeric_features", len(numeric_cols))
        mlflow.log_param("is_basarili", "Evet")
        
        # Artifact'ları kaydet
        mlflow.log_artifact(ARA_CIKTI_DOSYASI)
        mlflow.log_artifact(f"{ROBUST_SCALED_OUTPUT}_data.csv")
        
        elapsed_time = time.time() - start_time
        mlflow.log_metric("execution_time_seconds", elapsed_time)
        
        print(f"\n✓ MLflow Run ID: {run_id} başarıyla tamamlandı!")
        print(f"✓ Execution Time: {elapsed_time:.2f} saniye")
        
        # --- 10. ÖZNİTELİKLER ---
        print("\n=== OLUŞTURULAN ÖZNİTELİKLER ===")
        for i, feat in enumerate(numeric_cols, 1):
            print(f"{i}. {feat}")

    except Exception as e:
        print(f"\n!!!! HATA !!!!")
        print(e)
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("hata_mesaji", str(e))
        sys.exit(1)
        
    finally:
        spark.stop()
        print("\nSpark Session kapatıldı.")