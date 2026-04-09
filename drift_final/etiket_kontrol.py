import numpy as np
from sklearn.datasets import load_svmlight_file
import os

# Dataset klasörünün yolunu tanımla
# Script 'drift_final' içinde, veri 'Dataset' klasörünün içinde.
filename = 'Dataset/batch1.dat'

if os.path.exists(filename):
    print(f"\n>>> Dosya bulundu: {filename}")
    try:
        # Veriyi yükle
        X, y = load_svmlight_file(filename, n_features=128)
        
        # Etiketleri analiz et
        etiketler = np.unique(y)
        unique, counts = np.unique(y, return_counts=True)
        dagilim = dict(zip(unique, counts))
        
        print("-" * 30)
        print(f"BULDUM! Veri setindeki sınıflar: {etiketler}")
        print(f"Sayısal Dağılım: {dagilim}")
        print("-" * 30)
        
        # Yorumla
        if len(etiketler) == 2:
            print("YORUM: Bu 'Binary Classification' (0 ve 1).")
            print("Genelde: 0 = Normal, 1 = Saldırı")
        else:
            print(f"YORUM: Bu 'Multi-class' ({len(etiketler)} farklı tür var).")
            
    except Exception as e:
        print(f"Veri okunurken hata oluştu: {e}")
else:
    print(f"HATA: '{filename}' bulunamadı! Lütfen Dataset klasörünü kontrol et.")