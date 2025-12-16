import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. VERİ YÜKLEME (Senin klasör ismin olan 'Dataset' kullanılıyor)
def load_batch(batch_id, data_folder='Dataset'):
    filename = f'{data_folder}/batch{batch_id}.dat'
    X, y = load_svmlight_file(filename, n_features=128)
    return X.toarray(), y

print("Modeller hazırlanıyor...")

# --- MODELLERİN HAZIRLANMASI ---
# Statik Model: Sadece en başta eğitilecek.
static_model = RandomForestClassifier(n_estimators=50, random_state=42)

# Adaptif Model: Her adımda güncellenecek.
adaptive_model = RandomForestClassifier(n_estimators=50, random_state=42)

# SONUÇ LİSTELERİ
static_accuracies = []
adaptive_accuracies = []
batches = range(1, 11)

# --- İLK EĞİTİM (BATCH 1) ---
print("Başlangıç eğitimi yapılıyor (Batch 1)...")
X_init, y_init = load_batch(1)

# İki modeli de başlangıçta Batch 1 ile eğitiyoruz
static_model.fit(X_init, y_init)
adaptive_model.fit(X_init, y_init)

# --- SİMÜLASYON DÖNGÜSÜ ---
print("\n--- Karşılaştırmalı Test Başlıyor ---")

for i in batches:
    # 1. Yeni veriyi yükle
    X_current, y_current = load_batch(i)
    
    # 2. STATİK MODEL Tahmini (Eski bilgiyle tahmin etmeye çalışır)
    y_pred_static = static_model.predict(X_current)
    acc_static = accuracy_score(y_current, y_pred_static)
    static_accuracies.append(acc_static)
    
    # 3. ADAPTİF MODEL Tahmini (Bir önceki batch'in bilgisiyle tahmin eder)
    y_pred_adaptive = adaptive_model.predict(X_current)
    acc_adaptive = accuracy_score(y_current, y_pred_adaptive)
    adaptive_accuracies.append(acc_adaptive)
    
    print(f"Batch {i} -> Statik: %{acc_static*100:.1f} | Adaptif: %{acc_adaptive*100:.1f}")
    
    # 4. KRİTİK ADIM: Adaptif modeli GÜNCELLE
    # Model şu anki veriyi (Batch i) öğrenir ki, Batch i+1 geldiğinde hazır olsun.
    adaptive_model.fit(X_current, y_current)

# --- SONUÇLARI GRAFİKLEME ---
plt.figure(figsize=(10, 6))

# Kırmızı Çizgi: Çakılan model
plt.plot(batches, static_accuracies, marker='o', color='red', linestyle='--', linewidth=2, label='Statik Model (Drift Var)')

# Yeşil Çizgi: Kurtarıcı model
plt.plot(batches, adaptive_accuracies, marker='s', color='green', linestyle='-', linewidth=2, label='Adaptif Model (Çözüm)')

plt.title('Çözüm Kanıtı: Statik vs Adaptif Model', fontsize=14)
plt.xlabel('Zaman (Batch ID)', fontsize=12)
plt.ylabel('Doğruluk (Accuracy)', fontsize=12)
plt.grid(True, alpha=0.6)
plt.legend()
plt.xticks(batches)

plt.show()

print(f"\nKANIT: Batch 10'da Statik Model %{static_accuracies[-1]*100:.1f} başarıdayken,")
print(f"Adaptif Model %{adaptive_accuracies[-1]*100:.1f} başarıda tutundu.")