import tensorflow as tf
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. MODEL YÜKLEME (Keras 3 Legacy Desteği ile)
print("--- Model Yükleme Aşaması ---")
model_path = 'kaydedilen_model/'

try:
    # Önce standart yüklemeyi dene
    vae_model = tf.keras.models.load_model(model_path)
    legacy_mode = False
    print("Model Keras formatında başarıyla yüklendi!")
except Exception as e:
    # Hata verirse Keras 3'ün önerdiği TFSMLayer (Legacy) yöntemine geç
    print(f"Bilgi: Standart yükleme başarısız oldu, Legacy format deneniyor...")
    vae_model = tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    legacy_mode = True
    print("Model Legacy (SavedModel) formatında başarıyla yüklendi!")

# 2. BAĞIMSIZ VERİ SETİNİ ÇEK (Network Anomaly)
print("\n--- Veri Seti İndirme Aşaması ---")
try:
    dataset = load_dataset("vansh11/Network_Anomaly_Detection", split="train")
    df_network = dataset.to_pandas()
    print(f"Hugging Face veri seti başarıyla indirildi. Satır sayısı: {len(df_network)}")
except Exception as e:
    print(f"Hata: Veri seti indirilemedi! İnternet bağlantını kontrol et. Detay: {e}")
    exit()

# 3. VERİ ÖN İŞLEME VE 128 BOYUTA UYARLAMA
print("\n--- Veri Ön İşleme ---")
# Sadece sayısal sütunları seç (VAE sayılarla çalışır)
numeric_df = df_network.select_dtypes(include=[np.number])
X_network = numeric_df.values

# Padding/Cropping (128 boyuta sabitleme)
if X_network.shape[1] < 128:
    padding = np.zeros((X_network.shape[0], 128 - X_network.shape[1]))
    X_network = np.hstack((X_network, padding))
else:
    X_network = X_network[:, :128]

# Normalizasyon (0-1 arasına çekme)
X_network = (X_network - np.min(X_network)) / (np.max(X_network) - np.min(X_network) + 1e-7)
X_network = X_network.astype('float32')

# 4. TAHMİN VE RECONSTRUCTION ANALİZİ
print("\n--- Analiz Başlatılıyor ---")
if not legacy_mode:
    reconstructions = vae_model.predict(X_network)
else:
    # Legacy modda model bir fonksiyon (Layer) gibi çağrılır
    predictions_dict = vae_model(X_network)
    # Çıktı sözlüğündeki ilk değeri al (Genelde 'output_0')
    reconstructions = predictions_dict[list(predictions_dict.keys())[0]].numpy()

# MSE (Yeniden Yapılandırma Hatası) Hesaplama
mse = np.mean(np.power(X_network - reconstructions, 2), axis=1)

# 5. GÖRSELLEŞTİRME VE RAPORLAMA
print("\n--- Sonuçlar Hazırlanıyor ---")
plt.figure(figsize=(10, 6))
plt.hist(mse, bins=50, color='purple', alpha=0.7, label='Network Verisi Hatası')
plt.axvline(x=0.1, color='r', linestyle='--', label='Gaz Eşik Değeri (0.1)')
plt.title("Cross-Domain Anomali Testi: Network Verisi Üzerindeki VAE Performansı")
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Örnek Sayısı")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"İşlem Tamam! Ortalama Yeniden Yapılandırma Hatası: {np.mean(mse):.6f}")
print("Bu grafik, modelinin hiç görmediği bir veriyi 'anomali' olarak nasıl algıladığını gösterir.")