import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import matplotlib.pyplot as plt
import seaborn as sns

# Grafik stilini ayarlayalım (Paylaştığın görseldeki gibi temiz bir görünüm için)
sns.set_theme(style="whitegrid")

# ==========================================
# 1. VERİ HAZIRLIĞI (DRIFT SİMÜLASYONU)
# ==========================================
np.random.seed(42)
torch.manual_seed(42)

# Temiz Eğitim Verisi
train_data = np.random.normal(loc=0.0, scale=1.0, size=(1000, 10)).astype(np.float32)

# Kaymış (Drifted) Test Verisi - Zamanla Değişen Dünya
test_data_drifted = np.random.normal(loc=0.0, scale=1.0, size=(100, 10)).astype(np.float32)
test_data_drifted[:, :5] += 2.5 # İlk 5 özelliğin dengesini bozduk

train_loader = DataLoader(TensorDataset(torch.tensor(train_data)), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(test_data_drifted)), batch_size=1, shuffle=False)

# ==========================================
# 2. MODEL TANIMLAMASI (TABULAR MAE)
# ==========================================
class TabularMAE(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, latent_dim=16):
        super(TabularMAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, mask_ratio=0.2):
        if self.training and mask_ratio > 0:
            mask = torch.rand_like(x) > mask_ratio
            x_masked = x * mask
        else:
            x_masked = x
        return self.decoder(self.encoder(x_masked))

# ==========================================
# 3. PRE-TRAINING (EĞİTİM)
# ==========================================
model = TabularMAE(input_dim=10)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

model.train()
for epoch in range(20):
    for batch in train_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = model(inputs, mask_ratio=0.3)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

# ==========================================
# 4. TEST-TIME ADAPTATION VE VERİ TOPLAMA
# ==========================================
tta_model = copy.deepcopy(model)
tta_optimizer = optim.Adam(tta_model.encoder.parameters(), lr=0.01)

losses_before_tta = []
losses_after_tta = []

# İlk 10 canlı veri akışını izleyelim ve grafiğe dökelim
for i, batch in enumerate(test_loader):
    if i >= 10: break
    single_input = batch[0]
    
    # TTA Öncesi Hata
    tta_model.eval()
    with torch.no_grad():
        pred_before = tta_model(single_input, mask_ratio=0.0)
        loss_before = criterion(pred_before, single_input).item()
        losses_before_tta.append(loss_before)
    
    # Test-Time Adaptation (3 Adım Optimizasyon)
    tta_model.train()
    for step in range(3):
        tta_optimizer.zero_grad()
        pred_step = tta_model(single_input, mask_ratio=0.2)
        loss_step = criterion(pred_step, single_input)
        loss_step.backward()
        tta_optimizer.step()
        
    # TTA Sonrası Hata
    tta_model.eval()
    with torch.no_grad():
        pred_after = tta_model(single_input, mask_ratio=0.0)
        loss_after = criterion(pred_after, single_input).item()
        losses_after_tta.append(loss_after)

# ==========================================
# 5. GÖRSELLEŞTİRME (SİZİN TARZINIZDA)
# ==========================================
plt.figure(figsize=(12, 6))

x_indexes = np.arange(1, 11)
bar_width = 0.35

# TTA Öncesi Barlar (Kırmızımsı / Somon rengi)
plt.bar(x_indexes - bar_width/2, losses_before_tta, bar_width, 
        label='Klasik Model Hatası (TTA Öncesi / Drifted)', color='#ff7f7f', alpha=0.9)

# TTA Sonrası Barlar (Yeşilimsi / Zeytin rengi)
plt.bar(x_indexes + bar_width/2, losses_after_tta, bar_width, 
        label='Adapte Olmuş Model Hatası (TTA Sonrası - Canlı Uyum)', color='#6b8e23', alpha=0.9)

plt.xlabel('Gelen Canlı Veri Sırası (Zaman Akışı)', fontsize=12)
plt.ylabel('Yeniden Yapılandırma Hatası (Reconstruction Loss)', fontsize=12)
plt.title('Veri Kaymasına (Drift) Karşı Test-Time Adaptation Performansı', fontsize=14, fontweight='bold')
plt.xticks(x_indexes)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()

# Grafiği kaydet
plt.savefig('tta_performance.png', dpi=300)
print("\nGrafik başarıyla 'tta_performance.png' adıyla kaydedildi!")