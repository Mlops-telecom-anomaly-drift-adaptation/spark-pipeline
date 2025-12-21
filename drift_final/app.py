from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import threading
from river.drift import ADWIN  # River kütüphanesi (Doğru olan)
import pandas as pd

app = Flask(__name__)

# ---------------------------------------------------------
# 1. MODEL YÜKLEME
# ---------------------------------------------------------
MODEL_PATH = 'model_adwin_final.pkl'
MODEL_VERSION = 'v1.0-Adaptive-River-ADWIN-MultiClass'

if os.path.exists(MODEL_PATH):
    print(f">>> '{MODEL_PATH}' yükleniyor...")
    try:
        model = joblib.load(MODEL_PATH)
        print(">>> ✅ Model başarıyla yüklendi! Canlı sisteme hazır.")
    except Exception as e:
        print(f">>> ❌ Model yükleme hatası: {e}")
        model = None
else:
    print(f">>> ⚠️ UYARI: '{MODEL_PATH}' bulunamadı! Lütfen önce analiz kodunu çalıştırın.")
    model = None

# ---------------------------------------------------------
# 2. ADAPTASYON VE DRİFT MEKANİZMASI
# ---------------------------------------------------------
# ADWIN Nesnesi (River kütüphanesi)
adwin_detector = ADWIN() 

# Kilit (Lock): Aynı anda model güncellemesini engellemek için
retrain_lock = threading.Lock()
is_training = False 

def check_and_adapt(features, true_label=None):
    """Drift kontrolü yapar (River kütüphanesine uygun)"""
    global model, is_training, adwin_detector

    # NOT: Canlı sistemde 'true_label' (gerçek sonuç) hemen gelmez.
    # Bu yüzden burası simülasyon amaçlıdır.
    
    if model and true_label is not None:
        prediction = model.predict(features)[0]
        # Hata oranı (0 = doğru, 1 = hata)
        error = 1 if prediction != true_label else 0 
        
        # --- RIVER GÜNCELLEMESİ ---
        adwin_detector.update(error)

        # Drift Tespiti
        if adwin_detector.drift_detected:
            print("!!! ADWIN: KRİTİK DRIFT TESPİT EDİLDİ! !!!")
            adwin_detector = ADWIN() # Sıfırla
            return "Drift Detected (Alert Sent)"
    
    return "Stable"

def retrain_model_async():
    """Arka planda yeniden eğitim simülasyonu"""
    global is_training
    with retrain_lock:
        if is_training: return
        is_training = True
        print("--- ⏳ Yeniden Eğitim Başladı (Simülasyon) ---")
        import time
        time.sleep(5) # Eğitimi simüle etmek için bekleme
        print("--- ✅ Yeniden Eğitim Tamamlandı ---")
        is_training = False

# ---------------------------------------------------------
# 3. ENDPOINTLER (UÇ NOKTALAR)
# ---------------------------------------------------------

@app.route('/')
def home():
    return f"<h3>MLOps Servisi Çalışıyor - {MODEL_VERSION} 🚀</h3>"

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
        
        # --- MULTI-CLASS GÜNCELLEMESİ ---
        # Sonuç 1, 2, 3, 4, 5, 6 olabilir.
        result_code = int(prediction[0])
        
        # Her sonuca genel bir etiket veriyoruz
        label = f"Sınıf {result_code} (Class {result_code})"

        return jsonify({
            "tahmin_kodu": result_code,
            "durum": label,
            "drift_status": drift_status,
            "versiyon": MODEL_VERSION
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# CI/CD Trigger (Jenkins burayı tetikler)
@app.route('/retrain', methods=['POST'])
def retrain_trigger():
    if is_training:
        return jsonify({"message": "Eğitim zaten sürüyor."}), 409

    print("--- 🔨 Jenkins/DevOps Tarafından Tetiklendi ---")
    threading.Thread(target=retrain_model_async).start() 
    return jsonify({"message": "Yeniden eğitim başlatıldı", "durum": "Started"})

# Kubernetes Health Check (Load Balancer buraya bakar)
@app.route('/health', methods=['GET'])
def health_check():
    if model is None:
        return jsonify({"status": "error", "message": "Model yok"}), 503
    if is_training:
        return jsonify({"status": "warning", "message": "Eğitim sürüyor"}), 200
    return jsonify({"status": "healthy", "version": MODEL_VERSION}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)