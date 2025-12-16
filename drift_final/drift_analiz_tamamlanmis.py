import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
# --- Yeni Kütüphane ---
from skmultiflow.drift_detection import ADWIN
# -----------------------

# Gereksiz uyarıları kapatalım
warnings.filterwarnings("ignore")

# ==========================================
# AYARLAR VE YARDIMCI FONKSİYONLAR
# ==========================================
DATA_FOLDER = 'Dataset'
N_FEATURES = 128
BATCHES = range(1, 11)

def load_batch(batch_id):
    """Belirtilen batch'i yükler ve dense array olarak döndürür."""
    filename = f'{DATA_FOLDER}/batch{batch_id}.dat'
    X, y = load_svmlight_file(filename, n_features=N_FEATURES)
    return X.toarray(), y

def plot_confusion_matrix(y_true, y_pred, title, ax):
    """Karmaşıklık matrisi çizer."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Gerçek Etiket')
    ax.set_xlabel('Tahmin Edilen')

# ==========================================
# 1. KORELASYON ANALİZİ (GRAFİK 1)
# ==========================================
print("--- 1. Korelasyon Analizi Hazırlanıyor ---")
X_b1, y_b1 = load_batch(1)
df_corr = pd.DataFrame(X_b1[:, :20]) 
plt.figure(figsize=(12, 10))
sns.heatmap(df_corr.corr(), annot=False, cmap='coolwarm', center=0)
plt.title('Grafik 1: Özellikler Arası Korelasyon Isı Haritası (İlk 20 Özellik - Batch 1)')
plt.show()
print("Korelasyon grafiği çizildi. ✅\n")


# ==========================================
# 2. MODELLERİN HAZIRLANMASI VE BAŞLANGIÇ EĞİTİMİ
# ==========================================
print("--- 2. Modeller Hazırlanıyor ve Yarış Başlıyor ---")

# --- 4 Modeli Tanımla ---
model_static = RandomForestClassifier(n_estimators=50, random_state=42)
model_cumulative = RandomForestClassifier(n_estimators=50, random_state=42)
model_window = RandomForestClassifier(n_estimators=50, random_state=42)
model_adwin = RandomForestClassifier(n_estimators=50, random_state=42) # YENİ MODEL

# Sonuçları saklamak için listeler
acc_static = []
acc_cumulative = []
acc_window = []
acc_adwin = [] # YENİ LİSTE
drift_status = [] # Drift durumunu saklamak için

# Kümülatif hafıza için veri saklama
X_history = X_b1.copy()
y_history = y_b1.copy()

# --- ADWIN Dedektörü (Başlangıç Ayarı) ---
# Bizim senaryomuzda, modelin yanıldığı kısmı (hata oranını) izleyeceğiz.
adwin_detector = ADWIN() 
adwin_drift_detected = False
# ------------------------------------------

# --- BAŞLANGIÇ EĞİTİMİ (BATCH 1) ---
print("Modeller Batch 1 ile eğitiliyor...")
model_static.fit(X_b1, y_b1)
model_cumulative.fit(X_history, y_history)
model_window.fit(X_b1, y_b1) 
model_adwin.fit(X_b1, y_b1) # YENİ MODEL EĞİTİMİ

# Karmaşıklık matrisleri için yer ayırt
fig_cm, axes_cm = plt.subplots(1, 3, figsize=(18, 5))

# ==========================================
# 3. SİMÜLASYON DÖNGÜSÜ (BATCH 1-10)
# ==========================================
for i in BATCHES:
    print(f"\n>>> Batch {i} İşleniyor <<<")
    X_current, y_current = load_batch(i)
    
    # --- TEST AŞAMASI ---
    y_pred_static = model_static.predict(X_current)
    acc_s = accuracy_score(y_current, y_pred_static)
    acc_static.append(acc_s)
    
    y_pred_cumulative = model_cumulative.predict(X_current)
    acc_c = accuracy_score(y_current, y_pred_cumulative)
    acc_cumulative.append(acc_c)
    
    y_pred_window = model_window.predict(X_current)
    acc_w = accuracy_score(y_current, y_pred_window)
    acc_window.append(acc_w)
    
    # --- ADWIN Model Testi ---
    y_pred_adwin = model_adwin.predict(X_current)
    acc_a = accuracy_score(y_current, y_pred_adwin)
    acc_adwin.append(acc_a)
    
    print(f"Doğruluklar -> Statik: %{acc_s*100:.1f} | Kümülatif: %{acc_c*100:.1f} | Window: %{acc_w*100:.1f} | ADWIN (Akıllı): %{acc_a*100:.1f}")

    # --- ADWIN Drift Tespiti ve Adaptasyon AŞAMASI (YENİ) ---
    is_drift = False
    if i > 1:
        # 1. Modelin Hata Oranını (0=Doğru, 1=Hata) ADWIN'e besle
        errors = (y_pred_adwin != y_current).astype(int)
        for error in errors:
            adwin_detector.add_element(error)
            if adwin_detector.detected_change():
                is_drift = True
                print(f"!!! ADWIN: Batch {i} 'de Drift Tespit Edildi! Adaptasyon yapılıyor. !!!")
                break
    
    drift_status.append(1 if is_drift else 0) # Grafikte kullanmak için durumu sakla

    # --- ADAPTASYON (EĞİTİM) AŞAMASI ---
    if i < 10:
        # Kümülatif: Geçmişe ekle ve hepsini yeniden eğit
        X_history = np.vstack((X_history, X_current))
        y_history = np.hstack((y_history, y_current))
        model_cumulative.fit(X_history, y_history)
        
        # Window: Sadece şu anki veriyi kullanarak eğit
        model_window.fit(X_current, y_current)
        
        # ADWIN: Yalnızca Drift Tespit Edilirse Eğit
        if is_drift:
            # Buradaki adaptasyon metodunu seçebilirsiniz:
            # 1. Kayan Pencere (Sadece mevcut veriyi kullan)
            model_adwin.fit(X_current, y_current) 
            
            # 2. Kümülatif (Drift varsa, geçmişi de kullan)
            # model_adwin.fit(X_history, y_history)
            
            # Drift tespitinden sonra ADWIN penceresini sıfırla
            adwin_detector = ADWIN() 


    # --- Karmaşıklık Matrisleri İçin Örnekler Alma ---
    if i == 1:
        plot_confusion_matrix(y_current, y_pred_static, 'Grafik 4: Statik Model (Başlangıç - B1)', axes_cm[0])
    elif i == 8: # Driftin en yoğun olduğu yerlerden biri
        plot_confusion_matrix(y_current, y_pred_static, 'Grafik 5: Statik Model (Drift Anı - B8)', axes_cm[1])
        plot_confusion_matrix(y_current, y_pred_adwin, 'Grafik 6: ADWIN Model (Drift Anı - B8)', axes_cm[2]) # ADWIN ile değiştirildi

print("\nSimülasyon tamamlandı. Grafikler çiziliyor...")
plt.tight_layout()
plt.show() # CM grafiklerini göster

# ==========================================
# 4. SONUÇ GRAFİKLERİ
# ==========================================

# --- GRAFİK 2: Ana Karşılaştırma Grafiği ---
plt.figure(figsize=(12, 6))
plt.plot(BATCHES, acc_static, marker='o', color='red', linestyle='--', linewidth=2, label='1. Statik (Hiçbir Şey Yapma)')
plt.plot(BATCHES, acc_cumulative, marker='s', color='blue', linestyle='-', linewidth=2, label='2. Kümülatif (Hepsini Hatırla - Sürekli Adaptasyon)')
plt.plot(BATCHES, acc_window, marker='^', color='green', linestyle='-.', linewidth=2, label='3. Kayan Pencere (Sonuncuyu Hatırla - Sürekli Adaptasyon)')
plt.plot(BATCHES, acc_adwin, marker='d', color='purple', linestyle='-', linewidth=2, label='4. ADWIN (Akıllı Adaptasyon - Drift Varsa Eğit)') # YENİ ÇİZGİ
plt.title('Grafik 2: Adaptasyon Stratejilerinin Karşılaştırılması (Concept Drift)', fontsize=14)
plt.xlabel('Zaman (Batch ID)', fontsize=12)
plt.ylabel('Doğruluk (Accuracy)', fontsize=12)
plt.grid(True, alpha=0.6)
plt.xticks(BATCHES)
plt.legend()
plt.show()

# --- GRAFİK 3: Ortalama Başarı Karşılaştırması (Bar Chart) ---
avg_static = np.mean(acc_static)
avg_cumulative = np.mean(acc_cumulative)
avg_window = np.mean(acc_window)
avg_adwin = np.mean(acc_adwin) # YENİ ORTALAMA

plt.figure(figsize=(10, 5))
methods = ['Statik Model', 'Kümülatif', 'Kayan Pencere', 'ADWIN Adaptasyon'] # Yeni metot eklendi
avgs = [avg_static, avg_cumulative, avg_window, avg_adwin]
colors = ['red', 'blue', 'green', 'purple']
bars = plt.bar(methods, avgs, color=colors, alpha=0.7)
plt.title('Grafik 3: Yöntemlerin Ortalama Başarısı', fontsize=14)
plt.ylabel('Ortalama Doğruluk')
plt.ylim(0, 1.0)

for bar, avg in zip(bars, avgs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.05, f'%{avg*100:.1f}', 
             ha='center', color='white', fontweight='bold')
plt.show()

# --- GRAFİK 8: Drift Tespit Durumu (EK GRAFİK) ---
# Not: Bu grafik, ADWIN'in ne zaman alarm verdiğini gösterir.
plt.figure(figsize=(12, 3))
plt.bar(BATCHES, drift_status, color='orange', alpha=0.7)
plt.title('Grafik 8: ADWIN Drift Tespit Durumu', fontsize=14)
plt.xlabel('Batch ID')
plt.ylabel('Drift Tespit Edildi (1/0)')
plt.xticks(BATCHES)
plt.yticks([0, 1])
plt.grid(axis='y', alpha=0.5)
plt.show()

# --- GRAFİK 7: Veri Dağılımı Değişimi (Label Drift Kontrolü) ---
# Mevcut kodunuzdan alındı
label_counts = []
for i in BATCHES:
    _, y_batch = load_batch(i)
    counts = pd.Series(y_batch).value_counts(normalize=True).sort_index()
    label_counts.append(counts)

df_labels = pd.DataFrame(label_counts, index=BATCHES)
df_labels.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
plt.title('Grafik 7: Zaman İçinde Sınıf Dağılımı Değişimi (Label Drift Kontrolü)', fontsize=14)
plt.xlabel('Batch ID')
plt.ylabel('Oran (Proportion)')
plt.legend(title='Sınıflar (Etiketler)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- RAPORLAMA İÇİN SONUÇ METNİ ---
print("\n=== RAPOR İÇİN ÖZET ===")
print(f"Statik modelin minimum başarısı: %{min(acc_static)*100:.1f}")
print(f"Statik modelin ortalama başarısı: %{avg_static*100:.1f}")
print("--- Adaptasyon Metotları Karşılaştırması ---")
print(f"1. Kümülatif Adaptasyon (Ortalama): %{avg_cumulative*100:.1f}")
print(f"2. Kayan Pencere Adaptasyon (Ortalama): %{avg_window*100:.1f}")
print(f"3. ADWIN (Akıllı) Adaptasyon (Ortalama): %{avg_adwin*100:.1f}")
print("Grafikler, adaptasyon yöntemlerinin drift etkisini önemli ölçüde azalttığını göstermektedir.")