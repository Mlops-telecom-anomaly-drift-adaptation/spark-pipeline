import requests
import pandas as pd
import numpy as np
import time
import os

# API Uç Noktası
URL = "http://localhost:5000/predict"

# ---------------------------------------------------------
# 1. Veri Yükleme Fonksiyonu (LibSVM Temizliği Yapıldı)
# ---------------------------------------------------------
def load_batch(batch_number):
    """
    Belirtilen batch numarasını yükler ve LibSVM formatını (index:value) temizler.
    """
    # Klasör adınızın doğru olduğundan emin olun (Dataset veya dataset)
    file_path = f"Dataset/batch{batch_number}.dat" 
    
    if not os.path.exists(file_path):
        return None
        
    try:
        # Veriyi okurken boşlukları ayırıcı olarak kullan
        data = pd.read_csv(file_path, header=None, sep=' ')
        
        # Son sütun hariç tüm sütunları al (Özellikler X)
        X_raw = data.iloc[:, :-1]
        X_cleaned = X_raw.copy()
        
        # <<< LibSVM Formatından Temizleme >>>
        for col in X_raw.columns:
             # Değerleri string'e çevir, ilk ':' işaretinden sonrasını al
             X_cleaned[col] = X_raw[col].astype(str).str.split(':', n=1).str[-1]

        # Temizlenmiş veriyi numpy float array'e çevir
        X = X_cleaned.values.astype(float)
        return X
        
    except Exception as e:
        print(f"Veri Temizleme/Okuma Hatası ({file_path}): {e}")
        return None


# ---------------------------------------------------------
# 2. Toplu Test Başlatma
# ---------------------------------------------------------
MAX_BATCH_NUMBER = 20 # Veri setinizin kaç batch içerdiğini varsayıyoruz.

drift_detections = 0
total_requests = 0

print(">>> Kaydedilmiş Batch Dosyaları ile Uçtan Uca Test Başlatılıyor...")

for batch_num in range(1, MAX_BATCH_NUMBER + 1):
    X_batch = load_batch(batch_num)
    
    if X_batch is None:
        print(f"Batch {batch_num} atlandı (Dosya bulunamadı veya okunamadı).")
        continue

    print(f"\n--- Batch {batch_num} Yükleniyor ({len(X_batch)} örnek) ---")
    
    for i, features in enumerate(X_batch):
        # Veriyi app.py'ın beklediği [[...]] formatında gönderme
        payload = {"features": features.reshape(1, -1).tolist()} 
        
        try:
            response = requests.post(URL, json=payload, timeout=5)
            
            # API'den gelen hatalı cevabı yakala
            if response.status_code != 200:
                error_detail = response.json().get('error', f"HTTP Durumu: {response.status_code}")
                print(f"İstek HATA ({response.status_code}): {error_detail}")
                continue
                
            response_json = response.json()
            
            # Drift tespiti kontrolü
            if "Drift Detected" in response_json.get("drift_status", "Stable"):
                drift_detections += 1
                print(f"[!] DRIFT TESPİT EDİLDİ: Batch {batch_num}, Örnek {i}. Tahmin: {response_json['prediction']}")
            
            total_requests += 1

        except requests.exceptions.ConnectionError:
            print("\n!!! HATA: API sunucusu çalışmıyor. Lütfen birinci terminalde 'python app.py' çalıştırın. Çıkılıyor.")
            exit()
        except Exception as e:
            # Diğer genel hatalar
            pass
            
        # Tıkanmayı önlemek için minimum düzeyde gecikme (En son çözümümüz)
        time.sleep(0.001) 

print("\n--- TEST SONUÇLARI ---")
print(f"Gönderilen Toplam İstek: {total_requests}")
if drift_detections > 0:
    print(f"✅ KRİTİK BAŞARI: ADWIN, test sırasında {drift_detections} kez drift tespitini başarıyla tetikledi.")
    print("Bu, tüm sistemin (Veri Yükleme, Tahmin, ADWIN) çalıştığı anlamına gelir.")
else:
    print("Sonuç: Drift tespit edilmedi veya azdı.")