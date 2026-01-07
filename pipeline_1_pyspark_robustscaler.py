import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, window, concat, lit, stddev
from pyspark.ml.feature import RobustScaler, VectorAssembler
from pyspark.sql.types import DoubleType, IntegerType, LongType
import mlflow
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# --- 1. AYARLAR ---
BUYUK_VERI_SETI_YOLU = os.path.join(os.getcwd(), "TUBITAK_data_280925__041025_cleaned.csv")
OVERLAP_DOSYASI = os.path.join(os.getcwd(), "Overlap_matrix.csv")
OVERLAP_ESIGI = 40.0
PENCERE_SURESI = "8 hours"
ROBUST_SCALED_OUTPUT = "pipeline1_robust_scaled_pyspark"

# --- 2. MLflow & Prometheus ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pipeline 1 - RobustScaler PySpark")

PROMETHEUS_GATEWAY = "127.0.0.1:9091"
PROMETHEUS_JOB_NAME = 'pipeline1_robustscaler_pyspark'
registry = CollectorRegistry()
g_duration = Gauge('pipeline1_duration_seconds', 'Pipeline 1 Islem Suresi', registry=registry)
g_samples = Gauge('pipeline1_total_samples', 'Toplam Ornek Sayisi', registry=registry)

print("PIPELINE 1 (PySpark + Dot Fix) başlatılıyor...")
start_time = time.time()

spark = SparkSession.builder \
    .appName("Pipeline1_RobustScaler_PySpark") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"MLflow Run ID: {run_id} başladı.")
    
    try:
        mlflow.log_param("input_dataset", BUYUK_VERI_SETI_YOLU)
        
        # --- 4. VERİ YÜKLEME ---
        print("Veriler yükleniyor...")
        df_main = spark.read.csv(BUYUK_VERI_SETI_YOLU, header=True, inferSchema=True)
        df_overlap_raw = spark.read.csv(OVERLAP_DOSYASI, header=True, inferSchema=True)
        
        # --- 5. SÜTUN TEMİZLİĞİ ---
        new_columns = []
        for c in df_main.columns:
            # [H] siliniyor
            clean_c = c.strip().replace("[H] ", "")
            new_columns.append(clean_c)
            
        df_main = df_main.toDF(*new_columns)

        # Kritik İsim Düzeltmeleri
        if "SectorID" in df_main.columns: df_main = df_main.withColumnRenamed("SectorID", "CELL")
        if "SITE" in df_main.columns: df_main = df_main.withColumnRenamed("SITE", "CELL")
        
        # CELL birleştirme
        if "SITE" in df_main.columns and "FREQ" in df_main.columns:
             df_main = df_main.withColumn("CELL", concat(col("SITE").cast("string"), lit("_"), col("FREQ").cast("string")))

        # --- 6. OVERLAP & JOIN ---
        print("Overlap ve Join...")
        df_overlap_filtered = df_overlap_raw.filter(col("Overlap_Alan%") >= OVERLAP_ESIGI)
        df_joined = df_main.join(df_overlap_filtered.select("CELL", "N_CELL"), on="CELL", how="left")
        
        # --- 7. OTOMATİK NUMERIC SÜTUN SEÇİMİ ---
        numeric_cols = [f.name for f in df_joined.schema.fields 
                        if isinstance(f.dataType, (DoubleType, IntegerType, LongType)) 
                        and f.name not in ['LAT', 'LON', 'BW', 'CARRIER', 'AZIMUTH', 'FREQ']]
        
        print(f"Otomatik Algılanan Sayısal Sütunlar ({len(numeric_cols)} adet)")

        # --- 8. WINDOWING (VE NOKTA TEMİZLİĞİ) ---
        print("Windowing yapılıyor...")
        
        agg_exprs = []
        clean_numeric_cols = []
        
        for c in numeric_cols:
            # Sütun adındaki noktaları (.) alt çizgiye (_) çeviriyoruz!
            # Böylece "L.UL.Interference" -> "L_UL_Interference" oluyor.
            # Bu, Spark'ın kafa karışıklığını çözer.
            clean_name = c.replace(".", "_")
            new_col_name = f"avg_{clean_name}_8h"
            
            # Orijinal sütunu (backtick ile) alıp, yeni temiz isimle kaydediyoruz
            agg_exprs.append(avg(col(f"`{c}`")).alias(new_col_name))
            clean_numeric_cols.append(new_col_name)
        
        df_windowed = df_joined.groupBy(
            window(col("DATETIME"), PENCERE_SURESI),
            col("CELL"),
            col("N_CELL")
        ).agg(*agg_exprs).na.fill(0)
        
        total_rows = df_windowed.count()
        print(f"Windowed Veri Satır Sayısı: {total_rows}")
        mlflow.log_metric("total_samples", total_rows)

        # --- 9. ROBUST SCALER ---
        print("RobustScaler uygulanıyor...")
        
        # Artık temiz isimleri (noktasız) kullanıyoruz
        assembler = VectorAssembler(inputCols=clean_numeric_cols, outputCol="raw_features")
        df_assembled = assembler.transform(df_windowed)
        
        scaler = RobustScaler(inputCol="raw_features", outputCol="scaled_features", withCentering=True, withScaling=True)
        scaler_model = scaler.fit(df_assembled)
        df_scaled = scaler_model.transform(df_assembled)
        
        # --- 10. KAYDET ---
        print(f"Kaydediliyor: {ROBUST_SCALED_OUTPUT}")
        
        df_scaled.drop("raw_features").write.mode("overwrite").parquet(ROBUST_SCALED_OUTPUT)
        mlflow.log_artifact(ROBUST_SCALED_OUTPUT)
        
        elapsed = time.time() - start_time
        g_duration.set(elapsed)
        try:
            push_to_gateway(PROMETHEUS_GATEWAY, job=PROMETHEUS_JOB_NAME, registry=registry)
            print("✓ Prometheus metrikleri gönderildi")
        except:
            pass

        mlflow.log_metric("execution_time_seconds", elapsed)
        mlflow.log_param("is_basarili", "Evet")
        print(f"\n✅ BAŞARILI! Run ID: {run_id}")

    except Exception as e:
        print(f"\n❌ HATA: {e}")
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("error", str(e))
        sys.exit(1)
        
    finally:
        spark.stop()