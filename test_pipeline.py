#!/usr/bin/env python3
"""
TUBITAK Pipeline Test Script
Makine öğrenmesi pipeline'larını test et
"""

import os
import sys
import time
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_check(text):
    print(f"✓ {text}")

def print_error(text):
    print(f"✗ {text}")
    
def print_warning(text):
    print(f"⚠ {text}")

def print_info(text):
    print(f"ℹ {text}")

# ============================================
# 1. SISTEM KONTROL
# ============================================
print_header("SISTEM KONTROL")

# Python versiyon
py_version = sys.version.split()[0]
print(f"Python Versiyonu: {py_version}")

if sys.version_info < (3, 8):
    print_error("Python 3.8+ gerekli")
    sys.exit(1)
else:
    print_check("Python versiyonu uygun")

# ============================================
# 2. KÜTÜPHANE KONTROL
# ============================================
print_header("KÜTÜPHANE KONTROL")

REQUIRED_PACKAGES = {
    'pyspark': 'Apache Spark',
    'mlflow': 'MLflow',
    'prometheus_client': 'Prometheus Client'
}

MISSING_PACKAGES = {}

for package, name in REQUIRED_PACKAGES.items():
    try:
        __import__(package)
        print_check(f"{name} ({package}) kurulu")
    except ImportError:
        print_warning(f"{name} ({package}) bulunamadı")
        MISSING_PACKAGES[package] = name

if MISSING_PACKAGES:
    print_info("Eksik paketler:")
    for pkg, name in MISSING_PACKAGES.items():
        print(f"  - {name} ({pkg})")
    
    response = input("\nPacket'leri şimdi yükleme? (y/n): ")
    if response.lower() in ['y', 'yes']:
        import subprocess
        for pkg in MISSING_PACKAGES.keys():
            print(f"Yükleniyor: {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=False)
    else:
        print_error("Gerekli paketler yüklenmedi")
        sys.exit(1)
else:
    print_check("Tüm kütüphaneler mevcut")

# ============================================
# 3. DOSYA KONTROL
# ============================================
print_header("DOSYA KONTROL")

REQUIRED_FILES = {
    'pipeline_1_data_prep_full.py': 'Pipeline 1 (Windowing)',
    'pipeline_2_advanced_fs_full.py': 'Pipeline 2 (Feature Selection)',
    'config.py': 'Konfigürasyon dosyası'
}

MISSING_FILES = {}

for filename, description in REQUIRED_FILES.items():
    if Path(filename).exists():
        size = Path(filename).stat().st_size
        print_check(f"{description}: {filename} ({size} bytes)")
    else:
        print_warning(f"{description}: {filename} bulunamadı")
        MISSING_FILES[filename] = description

if MISSING_FILES:
    print_error("Eksik dosyalar:")
    for fname, desc in MISSING_FILES.items():
        print(f"  - {desc}: {fname}")

# ============================================
# 4. VERI DOSYALARI KONTROL
# ============================================
print_header("VERI DOSYALARI KONTROL")

data_files = {
    'TUBITAK_2807__030825.csv': 'Ana veri dosyası',
    'Overlap_matrix.csv': 'Overlap matrix dosyası'
}

for filename, description in data_files.items():
    if Path(filename).exists():
        size = Path(filename).stat().st_size
        size_mb = size / (1024 * 1024)
        print_check(f"{description}: {filename} ({size_mb:.2f} MB)")
    else:
        print_warning(f"{description}: {filename} bulunamadı")

# ============================================
# 5. SERVİS KONTROL
# ============================================
print_header("SERVİS KONTROL")

import socket

def check_port(host, port, service_name):
    """Port açık mı kontrol et"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            print_check(f"{service_name} ({host}:{port}) açık")
            return True
        else:
            print_warning(f"{service_name} ({host}:{port}) kapalı")
            return False
    except:
        print_warning(f"{service_name} ({host}:{port}) kontrol edilemiyor")
        return False

services = [
    ('localhost', 5000, 'MLflow'),
    ('localhost', 9000, 'MinIO'),
    ('localhost', 9091, 'Prometheus Pushgateway')
]

for host, port, name in services:
    check_port(host, port, name)

# ============================================
# 6. SPARK SESSION TEST
# ============================================
print_header("SPARK SESSION TEST")

try:
    from pyspark.sql import SparkSession
    
    print_info("Spark session başlatılıyor...")
    spark = SparkSession.builder \
        .appName("Test") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    # Test data oluştur
    test_data = [
        (1, "a", 10.0),
        (2, "b", 20.0),
        (3, "c", 30.0)
    ]
    
    df = spark.createDataFrame(test_data, ["id", "name", "value"])
    count = df.count()
    
    print_check(f"Spark session çalışıyor ({count} satır)")
    
    # RDD test
    rdd = spark.sparkContext.parallelize([1, 2, 3, 4, 5])
    sum_val = rdd.sum()
    print_check(f"RDD hesaplama çalışıyor (sum={sum_val})")
    
    spark.stop()
    print_check("Spark session kapatıldı")
    
except Exception as e:
    print_error(f"Spark test başarısız: {e}")
    sys.exit(1)

# ============================================
# 7. MLflow TEST
# ============================================
print_header("MLflow TEST")

try:
    import mlflow
    
    print_info("MLflow tracking URI set ediliyor...")
    mlflow.set_tracking_uri("http://localhost:5000")
    
    print_info("Test experiment oluşturuluyor...")
    mlflow.set_experiment("Test Experiment")
    
    print_info("Test run başlatılıyor...")
    with mlflow.start_run() as run:
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_metric", 42.0)
        
        run_id = run.info.run_id
        print_check(f"MLflow run oluşturuldu: {run_id}")
    
    print_check("MLflow çalışıyor")
    
except Exception as e:
    print_warning(f"MLflow test başarısız (server açık mı?): {e}")

# ============================================
# 8. CONFIG TEST
# ============================================
print_header("KONFIGÜRASYON TEST")

try:
    import config
    
    print_check(f"Data file: {config.BUYUK_VERI_SETI_YOLU}")
    print_check(f"Window size: {config.PENCERE_SURESI}")
    print_check(f"PCA k: {config.PCA_K_DEGERI}")
    print_check(f"Feature count: {len(config.SELECTED_FEATURES)}")
    
except Exception as e:
    print_warning(f"Config dosyası okunamadı: {e}")

# ============================================
# 9. ÇIKTI DİZİNLERİ KONTROL
# ============================================
print_header("ÇIKTI DİZİNLERİ")

output_dirs = {
    'windowed_features_output': 'Pipeline 1 çıktısı',
    'ml_ready_data_pca_parquet_full': 'Pipeline 2 çıktısı'
}

for dirname, description in output_dirs.items():
    if Path(dirname).exists():
        files = list(Path(dirname).glob("*"))
        print_check(f"{description}: {dirname} ({len(files)} dosya)")
    else:
        print_info(f"{description}: {dirname} (henüz oluşturulmadı)")

# ============================================
# 10. SÜRÜM BİLGİLERİ
# ============================================
print_header("SÜRÜM BİLGİLERİ")

try:
    from pyspark import __version__ as spark_version
    print(f"PySpark: {spark_version}")
except:
    print_warning("PySpark version kontrol edilemiyor")

try:
    import mlflow
    print(f"MLflow: {mlflow.__version__}")
except:
    print_warning("MLflow version kontrol edilemiyor")

try:
    import prometheus_client
    print(f"Prometheus Client: {prometheus_client.__version__}")
except:
    print_warning("Prometheus Client version kontrol edilemiyor")

# ============================================
# 11. ÖZET VE ÖNERİLER
# ============================================
print_header("ÖZET VE ÖNERİLER")

print("""
Pipeline sisteminizdeki durum:

Sistem Hazırlığı:
  ✓ Python 3.8+ kurulu
  ✓ Gerekli kütüphaneler mevcut
  ✓ Spark session çalışıyor
  
Sonraki Adımlar:
  1. Veri dosyalarını kontrol et (TUBITAK_2807__030825.csv, Overlap_matrix.csv)
  2. MLflow ve MinIO servislerini başlat (eğer kapalı ise)
  3. Pipeline'ları çalıştır:
     - python3 pipeline_1_data_prep_full.py
     - python3 pipeline_2_advanced_fs_full.py
  
Alternatif:
  - ./run_pipelines.sh TUBITAK_2807__030825.csv Overlap_matrix.csv

Monitoring:
  - MLflow: http://localhost:5000
  - Grafana: http://localhost:3000 (isteğe bağlı)

Yardım:
  - QUICK_START.md dosyasını oku
  - PIPELINE_METRICS_GUIDE.md dosyasını gözden geçir
""")

# ============================================
# 12. İNTERAKTİF MENU
# ============================================
print_header("İNTERAKTİF MENU")

while True:
    print("\nYapmak istediğin işlemi seç:")
    print("1. Pipeline 1'i çalıştır")
    print("2. Pipeline 2'yi çalıştır")
    print("3. Her iki pipeline'ı çalıştır")
    print("4. MLflow'u aç (http://localhost:5000)")
    print("5. Config'i göster")
    print("6. Çıkış")
    
    choice = input("\nSeçim (1-6): ").strip()
    
    if choice == "1":
        print_info("Pipeline 1 başlatılıyor...")
        os.system("python3 pipeline_1_data_prep_full.py")
    
    elif choice == "2":
        print_info("Pipeline 2 başlatılıyor...")
        os.system("python3 pipeline_2_advanced_fs_full.py")
    
    elif choice == "3":
        print_info("Pipeline 1 başlatılıyor...")
        ret1 = os.system("python3 pipeline_1_data_prep_full.py")
        if ret1 == 0:
            print_info("Pipeline 2 başlatılıyor...")
            os.system("python3 pipeline_2_advanced_fs_full.py")
        else:
            print_error("Pipeline 1 başarısız, Pipeline 2 atlanıyor")
    
    elif choice == "4":
        print_info("MLflow'u açmaya çalışıyor...")
        os.system("open http://localhost:5000 || xdg-open http://localhost:5000 || echo 'http://localhost:5000'")
    
    elif choice == "5":
        try:
            import config
            print("\nAktif Konfigürasyon:")
            print(f"  Input: {config.BUYUK_VERI_SETI_YOLU}")
            print(f"  Window: {config.PENCERE_SURESI}")
            print(f"  PCA k: {config.PCA_K_DEGERI}")
            print(f"  Features: {len(config.SELECTED_FEATURES)}")
            print(f"  MLflow: {config.MLFLOW_TRACKING_URI}")
        except Exception as e:
            print_error(f"Config okunamadı: {e}")
    
    elif choice == "6":
        print_check("Çıkılıyor...")
        sys.exit(0)
    
    else:
        print_warning("Geçersiz seçim")

print("\n" + "="*60)
print("Test tamamlandı!")
print("="*60)
