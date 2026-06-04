import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Az önce oluşturduğumuz src paketinden modeli çağırıyoruz
from src.model import TabularMAE

# MLflow Deney Adını Belirliyoruz
mlflow.set_experiment("Tabular_TTA_Data_Drift")

with mlflow.start_run():
    print("1. Temiz eğitim verileri üretiliyor...")
    np.random.seed(42)
    train_data = np.random.normal(loc=0.0, scale=1.0, size=(1000, 10)).astype(np.float32)
    train_loader = DataLoader(TensorDataset(torch.tensor(train_data)), batch_size=32, shuffle=True)

    # Hiperparametreler
    hidden_dim = 32
    latent_dim = 16
    lr = 0.005
    epochs = 20
    
    # MLOps Şovu: Parametreleri MLflow'a kaydet
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("latent_dim", latent_dim)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("epochs", epochs)

    # Modeli Kur
    model = TabularMAE(input_dim=10, hidden_dim=hidden_dim, latent_dim=latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("2. Model temiz veriyle eğitiliyor (Pre-training)...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs, mask_ratio=0.3)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # MLOps Şovu: Her epoch'taki hatayı MLflow grafiğine gönder
        epoch_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")

    # Eğitilen modeli doğrudan MLflow Model Registry'e fırlatıyoruz
    mlflow.pytorch.log_model(model, "tabular_mae_base_model")
    print("\n[BAŞARILI] Model eğitildi ve MLflow Model Registry'e kaydedildi!")