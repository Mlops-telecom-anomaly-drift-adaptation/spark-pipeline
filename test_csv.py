from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TestCSV").getOrCreate()

print("Test 1: Dosya kontrolü")
import os
if os.path.exists("TUBITAK_2807__030825.csv"):
    print("✓ Dosya var!")
    print(f"✓ Boyut: {os.path.getsize('TUBITAK_2807__030825.csv') / (1024**2):.2f} MB")
else:
    print("✗ Dosya yok!")

print("\nTest 2: CSV okuma")
try:
    df = spark.read.csv("TUBITAK_2807__030825.csv", header=True, inferSchema=True)
    print(f"✓ CSV okundu!")
    print(f"✓ Satır: {df.count()}")
    print(f"✓ Sütun: {len(df.columns)}")
    print(f"Sütunlar: {df.columns[:5]}")  # İlk 5
except Exception as e:
    print(f"✗ Hata: {e}")

spark.stop()