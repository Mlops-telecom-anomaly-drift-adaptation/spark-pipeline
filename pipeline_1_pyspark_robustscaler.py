import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, window
from pyspark.ml.feature import RobustScaler, VectorAssembler
import mlflow
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# --- 1. AYARLAR ---
BUYUK_VERI_SETI_YOLU = os.path.join(os.getcwd(), "TUBITAK_2807__030825.csv")
OVERLAP_DOSYASI = os.path.join(os.getcwd(), "Overlap_matrix.csv")
OVERLAP_ESIGI = 40.0
PENCERE_SURESI = "8 hours"
ARA_CIKTI_DOSYASI = "windowed_features_output"
ROBUST_SCALED_OUTPUT = "pipeline1_robust_scaled_pyspark"

# --- 2. MLflow & Prometheus ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pipeline 1 - RobustScaler PySpark")

# Prometheus
PROMETHEUS_GATEWAY = "127.0.0.1:9091"
PROMETHEUS_JOB_NAME = 'pipeline1_robustscaler_pyspark'
registry = CollectorRegistry()

g_duration = Gauge('pipeline1_duration_seconds', 'Pipeline 1 Işlem Suresi', registry=registry)
g_samples = Gauge('pipeline1_total_samples', 'Toplam Örnek Sayısı', registry=registry)
g_original_mean = Gauge('pipeline1_original_mean', 'Original Mean', registry=registry)
g_scaled_mean = Gauge('pipeline1_scaled_mean', 'Scaled Mean', registry=registry)

# --- 3. BAŞLAT ---
print("PIPELINE 1 (PySpark + RobustScaler) başlatılıyor...")
start_time = time.time()

spark = SparkSession.builder \
    .appName("Pipeline1_RobustScaler_PySpark") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    
    try:
        mlflow.log_param("input_dataset", BUYUK_VERI_SETI_YOLU)
        mlflow.log_param("scaler_type", "RobustScaler")
        mlflow.log_param("processing_framework", "PySpark")
        
        # --- 4. VERİ YÜKLEMESİ ---
        print(f"CSV dosyası okunuyor: {BUYUK_VERI_SETI_YOLU}")
        df_main = spark.read.csv(
            BUYUK_VERI_SETI_YOLU,
            header=True,
            inferSchema=True,
            timestampFormat="yyyy-MM-dd"
        )
        
        print(f"Overlap matris okunuyor: {OVERLAP_DOSYASI}")
        df_overlap_raw = spark.read.csv(OVERLAP_DOSYASI, header=True, inferSchema=True)
        
        # --- 5. SÜTUN TEMİZLİĞİ ---
        print("Sütun isimleri temizleniyor...")
        
        # Dinamik sütun temizleme
        for old_col in df_main.columns:
            new_col = old_col.strip()
            if new_col.startswith("[H] "):
                new_col = new_col.replace("[H] ", "")
            if old_col != new_col:
                df_main = df_main.withColumnRenamed(old_col, new_col)
        
        # CELL sütunu oluştur
        if "SITE" in df_main.columns and "FREQ" in df_main.columns:
            df_main = df_main.withColumn(
                "CELL",
                col("SITE").cast("string").concat(col("FREQ").cast("string"))
            )
        
        # --- 6. OVERLAP FİLTRELEME ---
        print(f"Overlap Eşiği (>= {OVERLAP_ESIGI}%) uygulanıyor...")
        df_overlap_filtered = df_overlap_raw.filter(col("Overlap_Alan%") >= OVERLAP_ESIGI)
        
        # --- 7. JOIN ---
        print("Join işlemi yapılıyor...")
        df_joined = df_main.join(
            df_overlap_filtered.select("CELL", "N_CELL"),
            on="CELL",
            how="left"
        )
        
        # --- 8. WINDOWING ---
        print("Windowing işlemi yapılıyor (8 saatlik pencereler)...")
        
        # Numeric sütunları belirle
        numeric_cols = [
            "Traffic_Volume_Gbyte",
            "ERAB_Drop_PC",
            "ERAB_ESTAB_ATT",
            "Nof_Avg_SimRRC_ConnUsr",
            "RRC_EstabSucc_PC",
            "NUM_OF_RRC_Att",
            "DLPRBUtilization",
            "DL_PRB_Util_%_HWI",
            "VOLTE_TRAFFIC_ERL",
            "AVGUL_RSSI_WEIGH_DBM_PUCCH",
            "AVGUL_RSSI_WEIGH_DBM_PUSCH",
            "Avg_UL_RSRP_PUCCH",
            "Avg_UL_RSRP_PUSCH",
            "Avg_CQI_HWI",
            "HO_Succ_PC_In"
        ]
        
        # Mevcut sütunları kontrol et
        available_cols = [col for col in numeric_cols if col in df_joined.columns]
        print(f"Kullanılacak numeric sütunlar: {len(available_cols)}")
        
        # Windowing aggregation
        df_windowed = df_joined.groupBy(
            window(col("DATETIME"), PENCERE_SURESI),
            col("CELL"),
            col("N_CELL")
        ).agg(
            *[avg(col(c)).alias(f"avg_{c}_8h") for c in available_cols]
        ).na.fill(0)
        
        print(f"Windowed veri satırları: {df_windowed.count()}")
        
        # --- 9. İSTATİSTİKLER ÖNCESİ ---
        print("\n=== ÖNCESİ (Original Data) ===")
        windowed_numeric_cols = [col for col in df_windowed.columns if "avg_" in col]
        
        stats_before = df_windowed.select(
            *[avg(col(c)).alias(f"mean_{i}") for i, c in enumerate(windowed_numeric_cols)]
        ).collect()[0]
        
        original_mean = sum(stats_before) / len(windowed_numeric_cols)
        print(f"Original Mean: {original_mean:.6f}")
        
        mlflow.log_metric("original_mean", float(original_mean))
        
        # --- 10. ROBUST SCALER (MLlib) ---
        print("\nRobustScaler uygulanıyor (PySpark MLlib)...")
        
        # VectorAssembler
        assembler = VectorAssembler(
            inputCols=windowed_numeric_cols,
            outputCol="raw_features"
        )
        df_assembled = assembler.transform(df_windowed)
        
        # RobustScaler
        scaler = RobustScaler(
            inputCol="raw_features",
            outputCol="scaled_features",
            withCentering=True,
            withScaling=True
        )
        scaler_model = scaler.fit(df_assembled)
        df_scaled = scaler_model.transform(df_assembled)
        
        # --- 11. İSTATİSTİKLER SONRASI ---
        print("\n=== SONRASI (After RobustScaler) ===")
        
        # Scaled features dönüştür
        from pyspark.sql.functions import col as spark_col
        from pyspark.ml.functions import vector_to_array
        
        df_scaled_array = df_scaled.withColumn(
            "scaled_array",
            vector_to_array(spark_col("scaled_features"))
        )
        
        # Mean hesapla
        scaled_stats = df_scaled_array.select(
            *[avg(spark_col("scaled_array")[i]).alias(f"mean_{i}") 
              for i in range(len(windowed_numeric_cols))]
        ).collect()[0]
        
        scaled_mean = sum(scaled_stats) / len(windowed_numeric_cols)
        print(f"Scaled Mean: {scaled_mean:.6f}")
        
        mlflow.log_metric("scaled_mean", float(scaled_mean))
        
        # --- 12. KAYDET ---
        print(f"\nParquet'e kaydediliyor: {ROBUST_SCALED_OUTPUT}")
        df_scaled.select(
            "window",
            "CELL",
            "N_CELL",
            "scaled_features"
        ).write.mode("overwrite").parquet(ROBUST_SCALED_OUTPUT)
        
        # --- 13. PROMETHEUS METRİKLERİ ---
        elapsed_time = time.time() - start_time
        total_samples = df_windowed.count()
        
        g_duration.set(elapsed_time)
        g_samples.set(total_samples)
        g_original_mean.set(float(original_mean))
        g_scaled_mean.set(float(scaled_mean))
        
        try:
            push_to_gateway(
                PROMETHEUS_GATEWAY,
                job=PROMETHEUS_JOB_NAME,
                registry=registry
            )
            print("✓ Prometheus metrikleri gönderildi")
        except:
            print("⚠️ Prometheus bağlantısı başarısız (devam ediliyor)")
        
        # --- 14. MLFLOW LOGGING ---
        mlflow.log_metric("total_samples", total_samples)
        mlflow.log_metric("numeric_features", len(available_cols))
        mlflow.log_metric("execution_time_seconds", elapsed_time)
        mlflow.log_param("windowing_interval", PENCERE_SURESI)
        mlflow.log_param("overlap_threshold", OVERLAP_ESIGI)
        
        print(f"\n✓ BAŞARILI!")
        print(f"✓ MLflow Run ID: {run_id}")
        print(f"✓ Execution Time: {elapsed_time:.2f} saniye")
        print(f"✓ Total Samples: {total_samples}")
        
    except Exception as e:
        print(f"\n✗ HATA: {e}")
        mlflow.log_param("error", str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        spark.stop()
        print("\nSpark Session kapatıldı.")