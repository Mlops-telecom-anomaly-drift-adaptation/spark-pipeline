from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import threading
from river.drift import ADWIN 
import pandas as pd

app = Flask(__name__)

# ---------------------------------------------------------
# 1. MODEL YÜKLEME
# ---------------------------------------------------------
MODEL_PATH = 'model_adwin_final.pkl'
MODEL_VERSION = 'v1.0-GasSensor-Drift-Adaptive'

if os.path.exists(MODEL_PATH):
    print(f">>> '{MODEL_PATH}' yükleniyor...")
    try:
        model = joblib.load(MODEL_PATH)
        print(">>> ✅ Model başarıyla yüklendi! Gaz Sensör Sistemi Aktif.")
    except Exception as e:
        print(f">>> ❌ Model yükleme hatası: {e}")
        model = None
else:
    print(f">>> ⚠️ UYARI: '{MODEL_PATH}' bulunamadı! Lütfen önce analiz kodunu çalıştırın.")
    model = None

# ---------------------------------------------------------
# 2. DRİFT MEKANİZMASI (Sensör Yaşlanma Takibi)
# ---------------------------------------------------------
adwin_detector = ADWIN() 
retrain_lock = threading.Lock()
is_training = False 

def check_and_adapt(features, true_label=None):
    """
    Drift (Sensör Kayması) kontrolü yapar.
    Sensörler zamanla kirlenir ve ölçümleri kayar (Concept Drift).
    """
    global model, is_training, adwin_detector
    
    if model and true_label is not None:
        prediction = model.predict(features)[0]
        error = 1 if prediction != true_label else 0 
        
        adwin_detector.update(error)

        if adwin_detector.drift_detected:
            print("!!! UYARI: Sensör Drifti Tespit Edildi! (Model Güncellenmeli) !!!")
            adwin_detector = ADWIN() # Dedektörü sıfırla
            return "Drift Detected (Sensor Aging)"
    
    return "Stable"

def retrain_model_async():
    """Arka planda yeniden eğitim simülasyonu"""
    global is_training
    with retrain_lock:
        if is_training: return
        is_training = True
        print("--- ⏳ Drift Adaptasyonu Başladı (Yeniden Eğitim) ---")
        import time
        time.sleep(5) 
        print("--- ✅ Model Güncellendi ve Sensör Kalibre Edildi ---")
        is_training = False

# ---------------------------------------------------------
# 3. ENDPOINTLER
# ---------------------------------------------------------

@app.route('/')
def home():
    return f"<h3>Gas Sensor Drift MLOps API - {MODEL_VERSION} 🧪</h3>"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model yüklü değil"}), 500

    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        # 1. Tahmin Yap
        prediction = model.predict(features)
        
        # 2. Drift Kontrolü (Simülasyon)
        drift_status = check_and_adapt(features, true_label=None)
        
        result_code = int(prediction[0])
        
        # --- BİLİMSEL ETİKETLEME (UCI Veri Setine Göre) ---
        # 1: Ethanol, 2: Ethylene, 3: Ammonia, 4: Acetaldehyde, 5: Acetone, 6: Toluene
        gas_names = {
            1: "Etanol (Ethanol)",
            2: "Etilen (Ethylene)",
            3: "Amonyak (Ammonia)",
            4: "Asetaldehit (Acetaldehyde)",
            5: "Aseton (Acetone)",
            6: "Toluen (Toluene)"
        }
        
        # Sözlükten ismi al, yoksa kodunu yaz
        label_text = gas_names.get(result_code, f"Bilinmeyen Gaz ({result_code})")

        return jsonify({
            "tahmin_kodu": result_code,
            "tespit_edilen_gaz": label_text, # API cevabı artık çok net!
            "sensor_durumu": drift_status,
            "versiyon": MODEL_VERSION
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# CI/CD Trigger (Jenkins burayı tetikler)
@app.route('/retrain', methods=['POST'])
def retrain_trigger():
    if is_training:
        return jsonify({"message": "Kalibrasyon zaten sürüyor."}), 409
    threading.Thread(target=retrain_model_async).start() 
    return jsonify({"message": "Sensör kalibrasyonu başlatıldı", "durum": "Started"})

# Kubernetes Health Check
@app.route('/health', methods=['GET'])
def health_check():
    if model is None:
        return jsonify({"status": "error", "message": "Model yok"}), 503
    return jsonify({"status": "healthy", "version": MODEL_VERSION}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)