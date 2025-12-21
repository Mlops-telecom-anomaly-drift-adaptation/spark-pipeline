import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import joblib 
import os

# --- DRİFT KÜTÜPHANESİ (RIVER) ---
from river.drift import ADWIN
# ---------------------------------

# Gereksiz uyarıları kapatalım
warnings.filterwarnings("ignore")

# ==========================================
# AYARLAR VE YARDIMCI FONKSİYONLAR
# ==========================================
DATA_FOLDER = 'Dataset'
N_FEATURES = 128
BATCHES = range(1, 11)

def load_batch(batch_id):
    """Veriyi yükler. Hem mevcut klasöre hem bir üst klasöre bakar."""
    # 1. Mevcut klasörde ara
    filename = f'{DATA_FOLDER}/batch{batch_id}.dat'
    if not os.path.exists(filename):
        # 2. Bulamazsan bir üst klasörde ara (Dataset/batch1.dat)
        filename = f'../{DATA_FOLDER}/batch{batch_id}.dat'
    
    try:
        X, y = load_svmlight_file(filename, n_features=N_FEATURES)
        return X.toarray(), y
    except FileNotFoundError:
        print(f"❌ HATA: {filename} dosyası bulunamadı! 'Dataset' klasörünün yerini kontrol et.")
        exit()

def plot_confusion_matrix(y_true, y_pred, title, ax):
    """Karmaşıklık matrisi çizer."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Gerçek')
    ax.set_xlabel('Tahmin')

# ==========================================
# 1. KORELASYON ANALİZİ
# ==========================================
print("--- 1. Korelasyon Analizi Hazırlanıyor ---")
X_b1, y_b1 = load_batch(1)
df_corr = pd.DataFrame(X_b1[:, :20]) 
plt.figure(figsize=(12, 10))
sns.heatmap(df_corr.corr(), annot=False, cmap='coolwarm', center=0)
plt.title('Grafik 1: Özellikler Arası Korelasyon (İlk 20 Özellik)')
plt.show()
print("Korelasyon grafiği çizildi. ✅\n")

# ==========================================
# 2. MODELLERİN HAZIRLANMASI
# ==========================================
print("--- 2. Modeller Hazırlanıyor ---")

model_static = RandomForestClassifier(n_estimators=50, random_state=42)
model_cumulative = RandomForestClassifier(n_estimators=50, random_state=42)
model_window = RandomForestClassifier(n_estimators=50, random_state=42)
model_adwin = RandomForestClassifier(n_estimators=50, random_state=42)

# Listeler
acc_static, acc_cumulative, acc_window, acc_adwin = [], [], [], []
drift_status = []

# Hafızalar
X_history = X_b1.copy()
y_history = y_b1.copy()

# --- ADWIN Dedektörü (River) ---
adwin_detector = ADWIN() 
# -------------------------------

# BAŞLANGIÇ EĞİTİMİ
print("Modeller Batch 1 ile eğitiliyor...")
model_static.fit(X_b1, y_b1)
model_cumulative.fit(X_history, y_history)
model_window.fit(X_b1, y_b1) 
model_adwin.fit(X_b1, y_b1)

fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))

# ==========================================
# 3. SİMÜLASYON DÖNGÜSÜ
# ==========================================
for i in BATCHES:
    print(f"\n>>> Batch {i} İşleniyor <<<")
    X_current, y_current = load_batch(i)
    
    # --- TAHMİNLER ---
    # Statik
    acc_s = accuracy_score(y_current, model_static.predict(X_current))
    acc_static.append(acc_s)
    
    # Kümülatif
    acc_c = accuracy_score(y_current, model_cumulative.predict(X_current))
    acc_cumulative.append(acc_c)
    
    # Window
    acc_w = accuracy_score(y_current, model_window.predict(X_current))
    acc_window.append(acc_w)
    
    # ADWIN
    y_pred_adwin = model_adwin.predict(X_current)
    acc_a = accuracy_score(y_current, y_pred_adwin)
    acc_adwin.append(acc_a)
    
    print(f"Başarılar -> Statik: %{acc_s*100:.1f} | Kümülatif: %{acc_c*100:.1f} | ADWIN: %{acc_a*100:.1f}")

    # --- DRİFT TESPİTİ (RIVER KODU DÜZELTİLDİ) ---
    is_drift = False
    if i > 1:
        errors = (y_pred_adwin != y_current).astype(int)
        for error in errors:
            # ESKİ: adwin_detector.add_element(error) -> HATALI
            # YENİ: adwin_detector.update(error) -> DOĞRU
            adwin_detector.update(error) 
            
            # ESKİ: if adwin_detector.detected_change(): -> HATALI
            # YENİ: if adwin_detector.drift_detected: -> DOĞRU
            if adwin_detector.drift_detected:
                is_drift = True
                print(f"!!! ADWIN UYARISI: Batch {i} 'de Drift Tespit Edildi! Model Güncelleniyor. !!!")
                break # Bu batch için drift bulundu, döngüden çık
    
    drift_status.append(1 if is_drift else 0)

    # --- ADAPTASYON (EĞİTİM) ---
    if i < 10:
        # Kümülatif (Hepsini eğit)
        X_history = np.vstack((X_history, X_current))
        y_history = np.hstack((y_history, y_current))
        model_cumulative.fit(X_history, y_history)
        
        # Window (Sadece yeniyi eğit)
        model_window.fit(X_current, y_current)
        
        # ADWIN (Sadece Drift Varsa eğit)
        if is_drift:
            model_adwin.fit(X_current, y_current) 
            adwin_detector = ADWIN() # Dedektörü sıfırla

    # --- Grafikler için Confusion Matrix ---
    if i == 1:
        plot_confusion_matrix(y_current, model_static.predict(X_current), 'Grafik 4: Statik (B1)', axes_cm[0])
    elif i == 8:
        plot_confusion_matrix(y_current, model_static.predict(X_current), 'Grafik 5: Statik (B8)', axes_cm[1])
        plot_confusion_matrix(y_current, y_pred_adwin, 'Grafik 6: ADWIN (B8)', axes_cm[2])

print("\nSimülasyon bitti. Grafikler açılıyor...")
plt.tight_layout()
plt.show()

# ==========================================
# 4. SONUÇ GRAFİKLERİ
# ==========================================

# Grafik 2: Karşılaştırma
plt.figure(figsize=(12, 6))
plt.plot(BATCHES, acc_static, 'r--o', label='Statik (Sabit)')
plt.plot(BATCHES, acc_cumulative, 'b-s', label='Kümülatif (Sürekli Öğrenen)')
plt.plot(BATCHES, acc_window, 'g-.^', label='Window (Unutan)')
plt.plot(BATCHES, acc_adwin, 'm-d', label='ADWIN (Drift Odaklı - Önerilen)')
plt.title('Grafik 2: Adaptasyon Stratejileri Karşılaştırması')
plt.xlabel('Zaman (Batch)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()
plt.show()

# Grafik 3: Ortalamalar
avgs = [np.mean(acc_static), np.mean(acc_cumulative), np.mean(acc_window), np.mean(acc_adwin)]
plt.figure(figsize=(10, 5))
bars = plt.bar(['Statik', 'Kümülatif', 'Window', 'ADWIN'], avgs, color=['red', 'blue', 'green', 'purple'])
plt.title('Grafik 3: Ortalama Başarı Oranları')
plt.ylim(0, 1.1)
for bar, avg in zip(bars, avgs):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'%{avg*100:.1f}', ha='center')
plt.show()

# Grafik 8: Drift Anları
plt.figure(figsize=(12, 3))
plt.bar(BATCHES, drift_status, color='orange')
plt.title('Grafik 8: ADWIN Drift Alarmları (1=Drift Var)')
plt.xlabel('Batch')
plt.yticks([0, 1])
plt.show()

# Grafik 7: Label Drift
label_counts = []
for i in BATCHES:
    _, y_batch = load_batch(i)
    counts = pd.Series(y_batch).value_counts(normalize=True).sort_index()
    label_counts.append(counts)
pd.DataFrame(label_counts, index=BATCHES).plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title('Grafik 7: Sınıf Dağılımı Değişimi')
plt.show()

# ==========================================
# 5. MLOPS: MODELİ KAYDETME
# ==========================================
print("\n[MLOps] Model canlı sistem için paketleniyor...")
joblib.dump(model_adwin, 'model_adwin_final.pkl') 
print("✅ Model başarıyla 'model_adwin_final.pkl' olarak kaydedildi.")
print("   -> Şimdi 'app.py' ve Docker aşamasına geçebilirsiniz.")