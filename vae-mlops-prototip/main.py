import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# 1. Veri Okuma Fonksiyonu
def veri_yukle(dosya_yolu):
    veriler = []
    try:
        with open(dosya_yolu, 'r') as f:
            for satir in f:
                parcalar = satir.split()
                ozellikler = [float(p.split(':')[1]) for p in parcalar[1:]]
                veriler.append(ozellikler)
        return np.array(veriler)
    except FileNotFoundError:
        print(f"Hata: {dosya_yolu} bulunamadı!")
        return None

# 2. VAE İçin Özel Örnekleme ve Kayıp Katmanı (Hata Çözümü Burada!)
class VAELayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        # Reparameterization trick
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        # KL Divergence Kaybı (Burada hesaplayarak hatayı önlüyoruz)
        kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        self.add_loss(tf.reduce_mean(kl_loss))
        return z

# --- ANA SÜREÇ ---

# A. Veri Hazırlığı
X_ham = veri_yukle('batch1.dat')
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_ham)
input_dim = X_train.shape[1]

# B. VAE Mimarisi
inputs = layers.Input(shape=(input_dim,))
h = layers.Dense(64, activation='relu')(inputs)
z_mean = layers.Dense(16)(h)
z_log_var = layers.Dense(16)(h)

# Özel katmanımızı çağırıyoruz (Hatayı çözen kısım)
z = VAELayer()([z_mean, z_log_var])

# Decoder
h_decoded = layers.Dense(64, activation='relu')(z)
outputs = layers.Dense(input_dim, activation='sigmoid')(h_decoded)

vae = keras.Model(inputs, outputs)
vae.compile(optimizer='adam', loss='mse') # MSE + VAELayer'dan gelen KL Loss

# C. Model Eğitimi
print("\n[INFO] VAE Modeli eğitiliyor...")
vae.fit(X_train, X_train, epochs=100, batch_size=16, verbose=0)

# D. Drift Testi (Batch 10)
X_drift_raw = veri_yukle('batch10.dat')
X_drift = scaler.transform(X_drift_raw)

X_train_pred = vae.predict(X_train)
train_mse = np.mean(np.square(X_train - X_train_pred), axis=1)

X_drift_pred = vae.predict(X_drift)
drift_mse = np.mean(np.square(X_drift - X_drift_pred), axis=1)

# E. Otomatik Retraining (Yeniden Eğitim)
threshold = np.mean(train_mse) * 1.2
print(f"\nBatch 1 Hata: {np.mean(train_mse):.6f} | Batch 10 Hata: {np.mean(drift_mse):.6f}")

if np.mean(drift_mse) > threshold:
    print("\n!!! [MLOps ALERT] Drift Tespit Edildi! Otomatik Adaptasyon Başlıyor...")
    vae.fit(X_drift, X_drift, epochs=20, verbose=0)
    X_post_pred = vae.predict(X_drift)
    post_mse = np.mean(np.square(X_drift - X_post_pred), axis=1)
    print(f"[SUCCESS] Adaptasyon Tamamlandı. Yeni Hata: {np.mean(post_mse):.6f}")

# F. Görselleştirme
plt.figure(figsize=(10, 6))
plt.hist(train_mse, bins=50, alpha=0.5, label='Normal (Batch 1)', color='blue')
plt.hist(drift_mse, bins=50, alpha=0.5, label='Drift (Batch 10 - Önce)', color='red')
if 'post_mse' in locals():
    plt.hist(post_mse, bins=50, alpha=0.5, label='Adaptasyon Sonrası', color='green')
plt.axvline(threshold, color='black', linestyle='--', label='Eşik')
plt.title('VAE Drift Tespiti ve Otonom Adaptasyon')
plt.legend()
plt.show()
