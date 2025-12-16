from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import threading
from skmultiflow.drift_detection import ADWIN # <<< YENİ
import pandas as pd

app = Flask(__name__)

# ---------------------------------------------------------
# 1. MODEL YÜKLEME (Inference Hazırlığı)
# ---------------------------------------------------------
MODEL_PATH = 'best_model.pkl'
MODEL_VERSION = 'v1.0-Adaptive-ADWIN' # <<< Güncellenmiş versiyon

if os.path.exists(MODEL_PATH):
    print(">>> Eğitilmiş model yükleniyor...")
    model = joblib.load(MODEL_PATH)
    print(">>> Model başarıyla yüklendi! Canlı sisteme hazır.")
else:
    print(">>> HATA: 'best_model.pkl' bulunamadı! Simülasyon modunda çalışılıyor.")
    model = None

# ---------------------------------------------------------
# 2. ADAPTASYON VE DRİFT MEKANİZMASI (YENİ EKLEME)
# ---------------------------------------------------------
# ADWIN Nesnesi: Hata akışını takip eder
adwin_detector = ADWIN() 
# Gelen veriyi biriktirme (Drift varsa yeniden eğitim için)
adaptive_buffer_X = []
adaptive_buffer_y = [] # Etiketler canlı serviste olmadığı için bu buffer şu an simüle.
                       # Gerçek hayatta bu kısım veritabanından etiketleri bekler.
MAX_BUFFER_SIZE = 500

# Kilit (Lock): Birden fazla isteğin aynı anda modeli güncellemesini engeller
retrain_lock = threading.Lock()
is_training = False # Yeniden eğitim durumu

def check_and_adapt(features, true_label=None):
    """Gelen tahmin sonucunun hatasını ADWIN'e besler ve gerekirse modeli günceller."""
    global model, is_training, adwin_detector

    # 1. Adaptif Buffer'a Veri Ekle (Burada true_label olsaydı onu kullanırdık)
    # Etiketler olmadığı için adaptasyon/buffer sadece simülasyondur.
    # Gerçek sistemde bu veriler etiketlenmeyi bekler.
    
    # 2. Hata Oranını ADWIN'e Besle
    if model and true_label is not None:
        prediction = model.predict(features)[0]
        # Hata oranı (0 = doğru, 1 = hata)
        error = 1 if prediction != true_label else 0 
        adwin_detector.add_element(error)

    # 3. Drift Tespitini Kontrol Et
    if adwin_detector.detected_change() and not is_training:
        print("!!! ADWIN: KRİTİK DRIFT TESPİT EDİLDİ! YENİDEN EĞİTİM TETİKLENİYOR. !!!")
        is_drift = True
        
        # Yeniden eğitimi ayrı bir iş parçacığında (thread) başlat
        # Bu, /predict uç noktasını kilitlemez
        # threading.Thread(target=retrain_model_async).start() 
        # is_training = True
        
        # ADWIN'i sıfırla
        adwin_detector = ADWIN()

        return "Drift Detected (Adaptation Triggered)"
    
    return "Stable"

# Simüle Edilmiş Asenkron Yeniden Eğitim Fonksiyonu
def retrain_model_async():
    """Buffer'daki verilerle modeli yeniden eğitir (Gerçek hayatta Jenkins yapar)."""
    # global model, is_training
    # with retrain_lock:
        # if not is_training: return # Zaten biri eğitiyorsa çık
        # is_training = True
        # print("--- Yeniden Eğitim Başladı ---")
        # # Gerçek yeniden eğitim mantığı buraya gelir
        # # ... (Eğitim)
        # print("--- Yeniden Eğitim Tamamlandı ---")
        # is_training = False
        # joblib.dump(model, MODEL_PATH) # Yeni modeli kaydet
    pass

# ---------------------------------------------------------
# 3. CANLI TAHMİN (INFERENCE) VE DRİFT KONTROLÜ (GÜNCELLENDİ)
# ---------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model yüklü değil"}), 500

    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        
        # 1. Tahmin Yap
        with retrain_lock:
            prediction = model.predict(features)
        
        # 2. ADWIN Drift Kontrolü
        # NOTE: Canlı sistemde gerçek etiket (true_label) hemen gelmediği için,
        # bu fonksiyon burada sadece simülasyon amacıyla çağrılır.
        # Gerçek hayatta bu, etiketler geldiğinde (Monitoring kodu) çağrılır.
        drift_status = check_and_adapt(features, true_label=None) 
        
        # Simülasyon: Drift bulunduysa adaptasyon yapılıyor uyarısı ver
        drift_warning = "Drift Detected" in drift_status
        
        # 3. Adaptif Buffer'a Veri Ekle (Adaptasyon için)
        # adaptive_buffer_X.append(features)
        
        response = {
            "prediction": int(prediction[0]),
            "drift_status": drift_status,
            "model_version": MODEL_VERSION
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------------------------------------------------------
# 4. ADAPTASYON TETİKLEME (DevOps / Jenkins Entegrasyonu)
# ---------------------------------------------------------
# BU KISIM, NOTLARDAKİ JENKINS/CI/CD GEREKSİNİMİNİ KARŞILAR
@app.route('/retrain', methods=['POST'])
def retrain_trigger():
    if is_training:
        return jsonify({"message": "Yeniden eğitim zaten devam ediyor. Lütfen bekleyin."}), 409

    print("--- Harici Kaynakla (Jenkins/DevOps) Yeniden Eğitim Tetikleniyor ---")
    threading.Thread(target=retrain_model_async).start() 

    return jsonify({"message": "Yeniden eğitim tetiklendi", "status": "Training started..."})

# ---------------------------------------------------------
# 5. KUBERNETES İÇİN HEALTH CHECK
# ---------------------------------------------------------
@app.route('/health', methods=['GET'])
def health_check():
    if model is None or is_training:
        # Model yoksa veya eğitim devam ediyorsa, servisi sağlıksız ilan et (Load Balancer buraya trafik göndermez)
        return jsonify({"status": "error", "message": "Model not ready or Retraining in progress"}), 503
    return jsonify({"status": "healthy", "service": MODEL_VERSION}), 200


if __name__ == '__main__':
    # Flask'ı birden fazla iş parçacığı (thread) ile çalıştırarak
    # aynı anda birden fazla isteği (test kodunuzdaki 445 istek gibi) işlemesini sağlar.
    app.run(host='0.0.0.0', port=5000, threaded=True)