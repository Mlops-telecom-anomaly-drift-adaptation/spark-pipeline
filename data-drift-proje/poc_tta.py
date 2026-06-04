import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

# ==========================================
# 1. VERİ HAZIRLIĞI (SENTETİK DRIFT SİMÜLASYONU)
# ==========================================
print("1. Aşama: Veri setleri üretiliyor...")
np.random.seed(42)
torch.manual_seed(42)

# Temiz Eğitim Verisi (Normal Dağılım)
train_data = np.random.normal(loc=0.0, scale=1.0, size=(1000, 10)).astype(np.float32)

# Kaymış (Drifted) Test Verisi (Non-stationary durum)
test_data_drifted = np.random.normal(loc=0.0, scale=1.0, size=(100, 10)).astype(np.float32)
test_data_drifted[:, :5] += 2.5 # İlk 5 özelliğin ortalamasını bozduk

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
# 3. PRE-TRAINING (MODELİN EĞİTİLMESİ)
# ==========================================
print("\n2. Aşama: Model temiz veriyle eğitiliyor (Pre-training)...")
model = TabularMAE(input_dim=10)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

model.train()
for epoch in range(20):
    total_loss = 0
    for batch in train_loader:
        inputs = batch[0]
        optimizer.zero_grad()
        outputs = model(inputs, mask_ratio=0.3)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 5 == 0:
        print(f"Epoch [{epoch+1}/20] - Ortalama Loss: {total_loss/len(train_loader):.4f}")

# ==========================================
# 4. TEST-TIME ADAPTATION (3DD-TTA VE TTT MANTIĞI)
# ==========================================
print("\n3. Aşama: Canlı akışta Test-Time Adaptation tetikleniyor...")

# Orijinal modeli bozmamak için kopyalıyoruz
tta_model = copy.deepcopy(model)
# TTA sırasında encoder ağırlıklarını güncelleyeceğimiz için optimizer tanımlıyoruz
tta_optimizer = optim.Adam(tta_model.encoder.parameters(), lr=0.01)

# Canlı sistemden sırayla 5 adet kaymış veri geldiğini varsayalım
for i, batch in enumerate(test_loader):
    if i >= 5: break
    single_input = batch[0]
    
    # TTA Öncesi Durum (Mevcut model veriyi ne kadar yabancıladı?)
    tta_model.eval()
    with torch.no_grad():
        pred_before = tta_model(single_input, mask_ratio=0.0)
        loss_before = criterion(pred_before, single_input).item()
    
    # Test-Time Optimization (Gelen tek bir satıra göre modeli 3 adım eğit)
    tta_model.train()
    for step in range(3):
        tta_optimizer.zero_grad()
        pred_step = tta_model(single_input, mask_ratio=0.2) # Maskeli adaptasyon
        loss_step = criterion(pred_step, single_input)
        loss_step.backward()
        tta_optimizer.step()
        
    # TTA Sonrası Durum (Model veriyi tanımaya başladı mı?)
    tta_model.eval()
    with torch.no_grad():
        pred_after = tta_model(single_input, mask_ratio=0.0)
        loss_after = criterion(pred_after, single_input).item()
        
    print(f"Canlı Veri {i+1} | TTA Öncesi Loss: {loss_before:.4f} ---> TTA Sonrası Loss: {loss_after:.4f}")

print("\nBaşarılı! Mekanizma tek betikte sorunsuz çalıştı.")