import streamlit as nn_st
import streamlit as st
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import time

st.set_page_config(page_title="Test-Time Adaptation MLOps Dashboard", layout="wide")

st.title("📊 Test-Time Adaptation & Data Drift İzleme Paneli")
st.markdown("Bu dashboard, MLflow'dan yüklenen TabularMAE modelinin canlı akış verilerine (Micro-batch) anlık olarak nasıl adapte olduğunu simüle eder.")

# ==========================================
# MODEL YÜKLEME
# ==========================================
@st.cache_resource
def load_mlflow_model():
    experiment = mlflow.get_experiment_by_name("Tabular_TTA_Data_Drift")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.train_loss ASC"])
    latest_run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{latest_run_id}/tabular_mae_base_model"
    return mlflow.pytorch.load_model(model_uri), latest_run_id

try:
    base_model, run_id = load_mlflow_model()
    st.success(f"✓ Aktif Model MLflow Registry'den Başarıyla Yüklendi! (Run ID: {run_id})")
except Exception as e:
    st.error("MLflow modeli yüklenemedi. Lütfen önce 'python train.py' çalıştırdığınızdan emin olun.")

# ==========================================
# CANLI AKIŞ BUTONU VE SİMÜLASYON
# ==========================================
if st.button("🚀 Canlı Veri Akışını ve TTA Mekanizmasını Başlat"):
    
    # Canlı veri üretimi (Drifted)
    np.random.seed(42)
    live_data_drifted = np.random.normal(loc=0.0, scale=1.0, size=(100, 10)).astype(np.float32)
    live_data_drifted[:, :5] += 2.5
    live_loader = DataLoader(TensorDataset(torch.tensor(live_data_drifted)), batch_size=10, shuffle=False)
    
    criterion = nn.MSELoss()
    
    losses_before = []
    losses_after = []
    
    # Canlı grafik alanları oluşturma
    chart_placeholder = st.empty()
    status_placeholder = st.empty()
    
    for batch_idx, batch in enumerate(live_loader):
        status_placeholder.info(f"⏳ Mikro-Batch {batch_idx+1}/10 işleniyor ve anlık TTA yapılıyor...")
        micro_batch = batch[0]
        
        tta_model = copy.deepcopy(base_model)
        tta_optimizer = optim.Adam(tta_model.encoder.parameters(), lr=0.01)
        
        # TTA Öncesi
        tta_model.eval()
        with torch.no_grad():
            pred_before = tta_model(micro_batch, mask_ratio=0.0)
            loss_before = criterion(pred_before, micro_batch).item()
            losses_before.append(loss_before)
            
        # TTA Adımları
        tta_model.train()
        for step in range(3):
            tta_optimizer.zero_grad()
            pred_step = tta_model(micro_batch, mask_ratio=0.2)
            loss_step = criterion(pred_step, micro_batch)
            loss_step.backward()
            tta_optimizer.step()
            
        # TTA Sonrası
        tta_model.eval()
        with torch.no_grad():
            pred_after = tta_model(micro_batch, mask_ratio=0.0)
            loss_after = criterion(pred_after, micro_batch).item()
            losses_after.append(loss_after)
            
        # SİZİN İSTEDİĞİNİZ TARZDA GRAPHİC SİMÜLASYONU (Her adımda güncellenir)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.set_theme(style="whitegrid")
        x_indexes = np.arange(1, len(losses_before) + 1)
        bar_width = 0.35
        
        ax.bar(x_indexes - bar_width/2, losses_before, bar_width, label='TTA Öncesi Hata (Klasik Model)', color='#ff7f7f')
        ax.bar(x_indexes + bar_width/2, losses_after, bar_width, label='TTA Sonrası Hata (Adapte Model)', color='#6b8e23')
        
        ax.set_xlabel('Gelen Canlı Mikro-Batch Sırası')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title('Canlı Akış Sırasında Test-Time Adaptation Performansı')
        ax.set_xticks(x_indexes)
        ax.legend()
        
        chart_placeholder.pyplot(fig)
        plt.close()
        
        # Canlı akış hissi vermek için kısa bir uyku süresi
        time.sleep(0.8)
        
    status_placeholder.success("🎯 Canlı veri akışı tamamlandı! TTA sayesinde operasyonel riskler başarıyla minimize edildi.")