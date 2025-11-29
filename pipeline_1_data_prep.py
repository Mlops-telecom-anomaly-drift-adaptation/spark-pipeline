import time
import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, window
import mlflow

# --- 1. Ayarlar ---
BUYUK_VERI_SETI_YOLU = os.path.join(os.getcwd(), "TUBITAK_2807__030825.csv")
OVERLAP_DOSYASI = os.path.join(os.getcwd(), "Overlap_matrix.csv")
OVERLAP_ESIGI = 40.0
PENCERE_SURESI = "8 hours"

# Bu pipeline'ın çıktısı olan ara dosyanın adı
ARA_CIKTI_DOSYASI = "windowed_features_output" 

# --- 2. MLflow & MinIO Sunucu Ayarları ---
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://127.0.0.1:9000'
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# MLflow'da yeni bir Deney (Experiment) adı veriyoruz
mlflow.set_experiment("Pipeline 1 - Veri Hazirlama (Windowing)")

# --- 3. Spark İşini Başlat ---
print(f"PIPELINE 1 (Veri Hazırlama) başlatılıyor...")
start_time = time.time()
spark = SparkSession.builder.appName("DataPrep_Windowing").getOrCreate()

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

        # --- 6. TRANSFORMASYONLAR (Hata Düzeltmeleri Dahil) ---

        # Hata düzeltme 1: Sütun adını düzelt ('SectorID' -> 'CELL')
        df_main = df_main.withColumnRenamed("SectorID", "CELL") 
        # Hata düzeltme 2: Sütun adını düzelt (Boşluk -> Alt çizgi)
        df_main = df_main.withColumnRenamed("Traffic Volume Gbyte", "Traffic_Volume_Gbyte")
        print(df_main.columns)
        print(f"Transformasyon: Overlap Eşiği (>= {OVERLAP_ESIGI}%) uygulanıyor...")
        df_overlap_filtered = df_overlap_raw.filter(col("Overlap_Alan%") >= OVERLAP_ESIGI)

        print("Ana veri ile filtrelenmiş Overlap verisi 'CELL' üzerinden birleştiriliyor...")
        df_joined = df_main.join(
            df_overlap_filtered.select("CELL", "N_CELL"),
            on="CELL",
            how="left"
        )

        print(f"Transformasyon: {PENCERE_SURESI}'lik pencereler halinde veriler gruplanıyor (Windowing)...")
        df_windowed = df_joined.groupBy(
            window(col("DATETIME"), PENCERE_SURESI),
            col("CELL"),
            col("N_CELL")
        ).agg(
            # ML için gerekli tüm öznitelikleri (feature) hesapla
            avg("Traffic_Volume_Gbyte").alias("avg_traffic_8h"),
            avg("ERAB_Drop_PC").alias("avg_erab_drop_rate_8h"),
            avg("Nof_Avg_SimRRC_ConnUsr").alias("avg_rrc_users_8h")
            # İhtiyaca göre buraya daha fazla .agg() eklenebilir
        )

        df_windowed = df_windowed.na.fill(0) # 'null' değerleri 0 ile doldur

        # --- 7. ARA ÇIKTIYI KAYDET ---
        print(f"Windowing sonucu ara dosya ({ARA_CIKTI_DOSYASI}) kaydediliyor...")
        # Veriyi Parquet formatında (CSV'den çok daha verimlidir) yerel diske kaydet
        df_windowed.write.mode("overwrite").parquet(ARA_CIKTI_DOSYASI)

        # Tüm 'windowed_features_output' klasörünü (Spark'ın çıktısı) MLflow'a (ve MinIO'ya) yükle
        mlflow.log_artifact(ARA_CIKTI_DOSYASI)

        mlflow.log_param("is_basarili", "Evet")
        print(f"MLflow Run ID: {run_id} başarıyla tamamlandı (Pipeline 1).")

    except Exception as e:
        print(f"!!!! PIPELINE 1'DE HATA OLUŞTU !!!!")
        print(e)
        mlflow.log_param("is_basarili", "Hayır")
        mlflow.log_param("hata_mesaji", str(e))
        sys.exit(1)

    finally:
        duration = time.time() - start_time
        print(f"İş (Pipeline 1) {duration:.2f} saniye sürdü.")
        mlflow.log_metric("job_duration_seconds", duration)

        # .count() işlemi bir eylem (action) gerektirdiği için 
        # ve zaten df_windowed.write.parquet() bir eylem olduğu için
        # sayımı yapmak için veriyi tekrar okumaya gerek yok, ama gerekirse:
        # df_count = spark.read.parquet(ARA_CIKTI_DOSYASI).count()
        # mlflow.log_metric("output_row_count", df_count)

        print("Spark Session durduruluyor (Pipeline 1).")
        spark.stop()
