import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Grafik ayarları
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 10})

# 1. VERİ YÜKLEME FONKSİYONU
def load_batch(batch_id, data_folder='Dataset'):
    filename = f'{data_folder}/batch{batch_id}.dat'
    try:
        X, y = load_svmlight_file(filename, n_features=128)
        return X.toarray(), y
    except FileNotFoundError:
        print(f"HATA: {filename} bulunamadı! Lütfen dosya yolunu kontrol edin.")
        return None, None

print(">>> SİSTEM BAŞLATILIYOR...")

# ---------------------------------------------------------
# AŞAMA 1: MODEL EĞİTİMİ VE 5-FOLD CROSS VALIDATION (KANITLAMA)
# ---------------------------------------------------------
print("\n[AŞAMA 1] Batch 1 Yükleniyor ve Model Doğrulanıyor (5-Fold CV)...")
X_init, y_init = load_batch(1)

# Modelleri Tanımla
# Static Model: Hiç güncellenmeyecek
model_static = RandomForestClassifier(n_estimators=50, random_state=42)
# Adaptive Model: Sürekli yeni veriyle eğitilecek
model_adaptive = RandomForestClassifier(n_estimators=50, random_state=42)

# 5'e Bölme (Hocanın İstediği Kısım)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_static, X_init, y_init, cv=cv, scoring='accuracy')

print(f"   -> 5-Fold Cross Validation Sonuçları: {cv_scores}")
print(f"   -> Ortalama Başarı: %{cv_scores.mean()*100:.2f} (Modelin sağlam olduğu kanıtlandı)")

# Modellerin ilk eğitimi
model_static.fit(X_init, y_init)
model_adaptive.fit(X_init, y_init) # İlk başta ikisi de aynı

# ---------------------------------------------------------
# AŞAMA 2: ZAMAN ARALIKLARINDA DRİFT VE ADAPTASYON TESTİ
# ---------------------------------------------------------
print("\n[AŞAMA 2] 10 Zaman Aralığı Boyunca Test ve Adaptasyon Başlıyor...")

# Sonuçları saklamak için listeler
batches = range(1, 11)
acc_static_list = []
acc_adaptive_list = []
f1_static_list = []
f1_adaptive_list = []
class_distributions = []

# Adaptasyon için kümülatif veri (Hafıza)
X_cumulative = X_init.copy()
y_cumulative = y_init.copy()

for i in batches:
    print(f"   -> İşleniyor: Batch {i}...")
    X_test, y_test = load_batch(i)
    
    # Sınıf dağılımını kaydet (Grafik için)
    unique, counts = np.unique(y_test, return_counts=True)
    dist_dict = dict(zip(unique, counts))
    class_distributions.append(dist_dict)
    
    # 1. STATIC MODEL TESTİ (Eski bilgiyle tahmin et)
    y_pred_static = model_static.predict(X_test)
    acc_static = accuracy_score(y_test, y_pred_static)
    f1_static = f1_score(y_test, y_pred_static, average='weighted')
    acc_static_list.append(acc_static)
    f1_static_list.append(f1_static)
    
    # 2. ADAPTIVE MODEL TESTİ (Güncel bilgiyle tahmin et)
    # Not: Gerçek hayatta önce tahmin eder, sonra etiketleri öğrenip eğitiriz.
    y_pred_adaptive = model_adaptive.predict(X_test)
    acc_adaptive = accuracy_score(y_test, y_pred_adaptive)
    f1_adaptive = f1_score(y_test, y_pred_adaptive, average='weighted')
    acc_adaptive_list.append(acc_adaptive)
    f1_adaptive_list.append(f1_adaptive)
    
    # ADAPTASYON ADIMI: Adaptive model yeni veriyi öğrenir (Retraining)
    # Batch 1 zaten eğitilmişti, Batch 2 ve sonrasında veri setini büyütüp yeniden eğitiyoruz
    if i < 10: # Son batch'ten sonra eğitmeye gerek yok
        X_next, y_next = load_batch(i+1) # Gelecek veriyi şimdiden alıp hazırlık yapmıyoruz, simülasyon gereği:
        # Mevcut veriyi hafızaya ekle ve modeli güncelle
        X_cumulative = np.vstack((X_cumulative, X_test))
        y_cumulative = np.hstack((y_cumulative, y_test))
        model_adaptive.fit(X_cumulative, y_cumulative) 

# ---------------------------------------------------------
# AŞAMA 3: KANITLAMA VE GRAFİKLER (EN AZ 7 GRAFİK)
# ---------------------------------------------------------
print("\n[AŞAMA 3] Grafikler Oluşturuluyor...")

fig = plt.figure(figsize=(18, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# GRAFİK 1: Accuracy Karşılaştırması (En Önemli Kanıt)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(batches, acc_static_list, 'r-o', label='Adaptasyonsuz (Static)', linewidth=2)
ax1.plot(batches, acc_adaptive_list, 'g-o', label='Adaptasyonlu (Retrain)', linewidth=2)
ax1.set_title('Accuracy: Adaptasyon vs Static')
ax1.set_xlabel('Zaman (Batch)')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# GRAFİK 2: F1 Score Karşılaştırması
ax2 = plt.subplot(2, 3, 2)
ax2.plot(batches, f1_static_list, 'r--', label='Static F1')
ax2.plot(batches, f1_adaptive_list, 'g--', label='Adaptive F1')
ax2.set_title('F1 Score Değişimi')
ax2.legend()
ax2.grid(True)

# GRAFİK 3: Drift Tespiti (Performans Kaybı)
# Static modelin ne kadar hata yaptığını gösterir
ax3 = plt.subplot(2, 3, 3)
error_rates = [1-x for x in acc_static_list]
ax3.bar(batches, error_rates, color='orange', alpha=0.7)
ax3.set_title('Static Model Hata Oranı (Drift Sinyali)')
ax3.set_ylabel('Hata Oranı (1 - Accuracy)')

# GRAFİK 4: Sınıf Dağılımı Değişimi (Class Distribution)
ax4 = plt.subplot(2, 3, 4)
df_dist = pd.DataFrame(class_distributions).fillna(0)
df_dist.plot(kind='bar', stacked=True, ax=ax4, colormap='viridis')
ax4.set_title('Zamanla Sınıf Dağılımı (Data Drift Sebebi?)')
ax4.set_xlabel('Batch ID')
ax4.legend(title='Class', fontsize='small')

# GRAFİK 5: Korelasyon Matrisi (Sadece İlk Batch)
# Özelliklerin birbiriyle ilişkisi
ax5 = plt.subplot(2, 3, 5)
df_batch1 = pd.DataFrame(X_init[:, :10]) # İlk 10 özellik için örnek
sns.heatmap(df_batch1.corr(), ax=ax5, cmap='coolwarm', cbar=False)
ax5.set_title('Özellik Korelasyonu (İlk 10 Feature)')

# GRAFİK 6: Confusion Matrix (Son Batch - Static Model)
# Modelin nerede yanıldığını görmek için
ax6 = plt.subplot(2, 3, 6)
cm = confusion_matrix(y_test, y_pred_static)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax6, cbar=False)
ax6.set_title('Confusion Matrix (Batch 10 - Static)')
ax6.set_ylabel('Gerçek')
ax6.set_xlabel('Tahmin')

plt.suptitle(f"DATA DRIFT VE ADAPTASYON ANALİZ RAPORU\nStatic Düşüş: %{acc_static_list[0]*100:.1f} -> %{acc_static_list[-1]*100:.1f} | Adaptive Koruma: %{acc_adaptive_list[-1]*100:.1f}", fontsize=16)
plt.show()

# GRAFİK 7: Feature Importance Değişimi (Ayrı Pencere)
# Başlangıçtaki önemli özellikler ile sondaki önemli özellikler değişti mi?
plt.figure(figsize=(10, 5))
importances_start = model_static.feature_importances_
importances_end = model_adaptive.feature_importances_
indices = np.argsort(importances_start)[::-1][:10] # En önemli 10 özellik

plt.bar(range(10), importances_start[indices], color='r', alpha=0.5, label='Batch 1 Önemi')
plt.bar(range(10), importances_end[indices], color='g', alpha=0.5, label='Batch 10 Önemi (Adaptive)')
plt.xticks(range(10), indices)
plt.title("En Önemli 10 Özelliğin Zamanla Değişimi (Feature Drift)")
plt.legend()
plt.show()

print("\nANALİZ TAMAMLANDI.")
print(f"Static Model Kaybı: %{(acc_static_list[0] - acc_static_list[-1])*100:.2f}")
print(f"Adaptasyon Kazancı: %{(acc_adaptive_list[-1] - acc_static_list[-1])*100:.2f}")

# ... (Kodun en alt kısmı) ...

import joblib

# Eğitilmiş en son adaptif modeli kaydediyoruz
print("Model canlı sistem için kaydediliyor...")
joblib.dump(model_adaptive, 'best_model.pkl')
print("Model başarıyla 'best_model.pkl' olarak kaydedildi! ✅")