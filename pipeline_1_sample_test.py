import time
import os
from sklearn.preprocessing import RobustScaler
import pandas as pd
import mlflow
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pipeline 1 - RobustScaler Sample Test")

print("PIPELINE 1 (RobustScaler - Sample) başlatılıyor...")
start_time = time.time()

with mlflow.start_run() as run:
    try:
        # Sample veri oluştur
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Traffic_Volume': np.random.normal(100, 50, n_samples),
            'ERAB_Drop_Rate': np.random.normal(5, 2, n_samples),
            'RRC_Users': np.random.normal(500, 100, n_samples),
            'DL_PRB_Util': np.random.normal(75, 15, n_samples),
            'CQI': np.random.normal(8.5, 1.5, n_samples),
        }
        
        df_pandas = pd.DataFrame(data)
        
        print(f"\n=== ÖNCESİ (Original) ===")
        print(f"Satır: {len(df_pandas)}")
        print(f"Sütun: {len(df_pandas.columns)}")
        print(f"Mean: {df_pandas.mean().mean():.4f}")
        print(f"Std: {df_pandas.std().mean():.4f}")
        print(f"Median: {df_pandas.median().mean():.4f}")
        
        # RobustScaler uygula
        scaler = RobustScaler()
        df_scaled = df_pandas.copy()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_pandas),
            columns=df_pandas.columns
        )
        
        print(f"\n=== SONRASI (After RobustScaler) ===")
        print(f"Mean: {df_scaled.mean().mean():.4f}")
        print(f"Std: {df_scaled.std().mean():.4f}")
        print(f"Median: {df_scaled.median().mean():.4f}")
        
        # Kaydet
        df_scaled.to_csv('pipeline1_robust_scaled_sample.csv', index=False)
        df_scaled.to_parquet('pipeline1_robust_scaled_sample.parquet')
        
        # MLflow log
        mlflow.log_metric("original_mean", df_pandas.mean().mean())
        mlflow.log_metric("scaled_mean", df_scaled.mean().mean())
        mlflow.log_metric("samples", len(df_scaled))
        mlflow.log_param("scaler_type", "RobustScaler")
        
        elapsed = time.time() - start_time
        mlflow.log_metric("execution_time", elapsed)
        
        print(f"\n✓ Başarılı! ({elapsed:.2f}s)")
        print(f"✓ Dosyalar kaydedildi:")
        print(f"  - pipeline1_robust_scaled_sample.csv")
        print(f"  - pipeline1_robust_scaled_sample.parquet")
        
    except Exception as e:
        print(f"✗ Hata: {e}")
        mlflow.log_param("error", str(e))