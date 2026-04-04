import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import mlflow
import os

# --- HUGGING FACE GÜNCEL KÜTÜPHANELERİ ---
from huggingface_hub import login, HfApi

# 1. Hugging Face Girişi (Write yetkili token'ını buraya yapıştır)
login("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")

# 2. Deney Ayarları
curr_path = os.getcwd()
mlflow.set_tracking_uri(f"file:///{curr_path}/mlruns")
mlflow.set_experiment("gas-sensor-autoencoder")

input_dim = 128
X_train = np.random.rand(100, input_dim)
X_val   = np.random.rand(20,  input_dim)

# 3. VAE Model Mimarisi Oluşturma (karthik-2905 Referanslı)
inputs    = layers.Input(shape=(input_dim,))
h         = layers.Dense(64, activation='relu')(inputs)
z_mean    = layers.Dense(16)(h)
z_log_var = layers.Dense(16)(h)

def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=(tf.shape(z_mean)[0], 16))
    return z_mean + tf.exp(0.5 * z_log_var) * eps

z       = layers.Lambda(sampling)([z_mean, z_log_var])
h_dec   = layers.Dense(64, activation='relu')(z)
outputs = layers.Dense(input_dim, activation='sigmoid')(h_dec)

vae = keras.Model(inputs, outputs)
vae.compile(optimizer='adam', loss='mse')

# 4. Eğitim ve MLflow Takibi
with mlflow.start_run(run_name="vae-gas-sensor"):

    mlflow.log_params({
        "input_dim":  input_dim,
        "latent_dim": 16,
        "epochs":     20,
        "batch_size": 16
    })

    print("\n[INFO] VAE Modeli eğitiliyor...")
    history = vae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=20,
        batch_size=16,
        verbose=1
    )

    # Metrikleri Logla
    for i in range(len(history.history['loss'])):
        mlflow.log_metric("loss",     history.history['loss'][i],     step=i)
        mlflow.log_metric("val_loss", history.history['val_loss'][i], step=i)

    X_pred     = vae.predict(X_train, verbose=0)
    X_pred_val = vae.predict(X_val,   verbose=0)
    threshold  = 0.1

    mlflow.log_metric("reconstruction_accuracy",     float(np.mean(np.abs(X_pred - X_train) < threshold)))
    mlflow.log_metric("val_reconstruction_accuracy", float(np.mean(np.abs(X_pred_val - X_val) < threshold)))

    print("[SUCCESS] MLflow UI > vae-gas-sensor > Model metrics başarıyla kaydedildi.")

# 5. EĞİTİM BİTTİKTEN SONRA HUGGING FACE'E GÖNDERME (YENİ YÖNTEM)
model_id = "zeyneppkalkannn/gas-sensor-vae-drift" 
yerel_klasor = "kaydedilen_model"

try:
    print(f"\n[MLOps] Model Hugging Face Hub için paketleniyor...")
    
    # Adım A: Modeli Keras'ın yeni formatına uygun olarak 'export' et
    vae.export(yerel_klasor) 
    
    # Adım B: O klasörü API ile Hugging Face sayfana gönder
    print(f"[MLOps] Hugging Face sunucularına aktarılıyor: {model_id}")
    api = HfApi()
    
    # Eğer repo henüz yoksa otomatik oluşturur
    api.create_repo(repo_id=model_id, exist_ok=True) 
    
    api.upload_folder(
        folder_path=yerel_klasor,
        repo_id=model_id,
        repo_type="model",
    )
    print("\n[SUCCESS] HARİKA! Model artık Hugging Face üzerinde erişilebilir durumda!")
except Exception as e:
    print(f"\n[ERROR] Hugging Face yükleme hatası: {e}")