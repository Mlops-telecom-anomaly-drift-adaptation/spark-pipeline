import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy

# ==========================================
# 1. MODELİ MLFLOW REGISTRY'DEN GERİ YÜKLE
# ==========================================
print("1. Aşama: En son eğitilen base model MLflow'dan indiriliyor...")
# MLflow'dan modeli yüklemek için bir sorgu çalıştırıyoruz
experiment = mlflow.get_experiment_by_name("Tabular_TTA_Data_Drift")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.train_loss ASC"])
latest_run_id = runs.iloc[0]["run_id"]

# MLflow deposundaki modelin adresi
model_uri = f"runs:/{latest_run_id}/tabular_mae_base_model"
base_model = mlflow.pytorch.load_model(model_uri)
base_model.eval()
print(f"[YÜKLENDİ] Model Run ID: {latest_run_id} başarıyla geri çağrıldı.\n")

# ==========================================
# 2. CANLI VERİ AKIŞI SİMÜLASYONU (MICRO-BATCHING MANTIĞI)
# ==========================================
print("2. Aşama: Canlı akış verileri hazırlanıyor...")
np.random.seed(42)

# Gerçek hayat simülasyonu: Dünyanın dengesi bozuldu (Data Drift)
# Canlı sisteme 100 satırlık kaymış veri geliyor
live_data_drifted = np.random.normal(loc=0.0, scale=1.0, size=(100, 10)).astype(np.float32)
live_data_drifted[:, :5] += 2.5 # İlk 5 kolon bozuk/kaymış durumda

# GÜRÜLTÜDEN KAÇIŞ: batch_size=10 yaparak mikro-batch topluyoruz!
live_loader = DataLoader(TensorDataset(torch.tensor(live_data_drifted)), batch_size=10, shuffle=False)

# ==========================================
# 3. TEST-TIME ADAPTATION PIPELINE
# ==========================================
print("3. Aşama: Canlı mikro-batch akışı ve TTA optimizasyonu başlıyor...\n")

criterion = nn.MSELoss()

# Canlı sistemde her 10'arlı veri grubu geldikçe bu döngü dönecek
for batch_idx, batch in enumerate(live_loader):
    micro_batch = batch[0]
    
    # Her mikro-batch geldiğinde base modelin temiz bir kopyasını alıyoruz
    # Böylece model bir önceki gruptan kalan hatalarla zehirlenmiyor
    tta_model = copy.deepcopy(base_model)
    tta_optimizer = optim.Adam(tta_model.encoder.parameters(), lr=0.01)
    
    # 1. Adım: TTA Öncesi Durum (Model bu 10'lu grubu ilk gördüğünde ne kadar şok oldu?)
    tta_model.eval()
    with torch.no_grad():
        pred_before = tta_model(micro_batch, mask_ratio=0.0)
        loss_before = criterion(pred_before, micro_batch).item()
        
    # 2. Adım: Test-Time Adaptation (Bu 10'lu gruba göre modeli 3 adımcık eğit)
    # 10 veri bir arada olduğu için gürültülü tek bir satır modeli saptıramayacak!
    tta_model.train()
    for step in range(3):
        tta_optimizer.zero_grad()
        pred_step = tta_model(micro_batch, mask_ratio=0.2) # Maskeli adaptasyon
        loss_step = criterion(pred_step, micro_batch)
        loss_step.backward()
        tta_optimizer.step()
        
    # 3. Adım: TTA Sonrası Durum (Model 10'lu grubu inceleyip yeni dünyaya uyum sağladı mı?)
    tta_model.eval()
    with torch.no_grad():
        pred_after = tta_model(micro_batch, mask_ratio=0.0)
        loss_after = criterion(pred_after, micro_batch).item()
        
    print(f"Mikro-Batch {batch_idx+1} (10 Satır) | TTA Öncesi Loss: {loss_before:.4f} ---> TTA Sonrası Loss: {loss_after:.4f}")

print("\n[BAŞARILI] Tüm canlı akış mikro-batch'ler halinde işlendi ve TTA başarıyla uygulandı!")