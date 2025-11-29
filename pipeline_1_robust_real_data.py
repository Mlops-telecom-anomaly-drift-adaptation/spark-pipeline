import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import mlflow
import time

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pipeline 1 - RobustScaler Real Data Fixed")

print("PIPELINE 1 (RobustScaler - GERÇEK VERİ) başlatılıyor...")
start_time = time.time()

with mlflow.start_run() as run:
    try:
        # CSV'i parça parça oku
        print("CSV dosyası okunuyor: TUBITAK_2807__030825.csv")
        df_list = []
        
        for i, chunk in enumerate(pd.read_csv('TUBITAK_2807__030825.csv', chunksize=10000)):
            df_list.append(chunk)
            if (i + 1) % 10 == 0:
                print(f"  ✓ {(i+1)*10000} satır okundu...")
        
        df_main = pd.concat(df_list, ignore_index=True)
        print(f"✓ Toplam {len(df_main)} satır yüklendi")
        
        # CELL sütunu oluştur (SITE + FREQ kombinasyonu)
        print("CELL sütunu oluşturuluyor...")
        df_main['CELL'] = df_main['SITE'].astype(str) + '_' + df_main['FREQ'].astype(str)
        
        # Overlap dosyasını oku
        print("Overlap matris okunuyor...")
        df_overlap = pd.read_csv('Overlap_matrix.csv')
        df_overlap_filtered = df_overlap[df_overlap['Overlap_Alan%'] >= 40.0]
        
        # Merge
        print("Join işlemi yapılıyor...")
        df_joined = df_main.merge(df_overlap_filtered[['CELL', 'N_CELL']], on='CELL', how='left')
        
        # Numeric sütunları seç
        numeric_cols = df_joined.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['LAT', 'LON', 'BW']]
        
        print(f"✓ Numeric sütun sayısı: {len(numeric_cols)}")
        
        # Windowing
        if 'DATETIME' in df_joined.columns:
            print("Windowing işlemi yapılıyor...")
            df_joined['DATETIME'] = pd.to_datetime(df_joined['DATETIME'], format='mixed')
            df_joined['time_window'] = df_joined['DATETIME'].dt.floor('8H')
            df_windowed = df_joined.groupby(['time_window', 'CELL'])[numeric_cols].mean().reset_index()
        else:
            df_windowed = df_joined[['CELL'] + numeric_cols]
        
        print(f"✓ Windowed veri: {len(df_windowed)} satır")
        
        # İstatistikler ÖNCESİ
        print("\n=== ÖNCESİ (Original Data) ===")
        original_mean = df_windowed[numeric_cols].mean().mean()
        original_std = df_windowed[numeric_cols].std().mean()
        original_median = df_windowed[numeric_cols].median().mean()
        
        print(f"Mean: {original_mean:.6f}")
        print(f"Std: {original_std:.6f}")
        print(f"Median: {original_median:.6f}")
        
        # RobustScaler
        print("\nRobustScaler uygulanıyor...")
        scaler = RobustScaler()
        df_windowed[numeric_cols] = scaler.fit_transform(df_windowed[numeric_cols])
        
        # İstatistikler SONRASI
        print("\n=== SONRASI (After RobustScaler) ===")
        scaled_mean = df_windowed[numeric_cols].mean().mean()
        scaled_std = df_windowed[numeric_cols].std().mean()
        scaled_median = df_windowed[numeric_cols].median().mean()
        
        print(f"Mean: {scaled_mean:.6f}")
        print(f"Std: {scaled_std:.6f}")
        print(f"Median: {scaled_median:.6f}")
        
        # Kaydet
        print("\nDosyalar kaydediliyor...")
        df_windowed.to_csv('pipeline1_robust_scaled_real.csv', index=False)
        df_windowed.to_parquet('pipeline1_robust_scaled_real.parquet')
        
        # MLflow
        mlflow.log_metric("original_mean", original_mean)
        mlflow.log_metric("original_std", original_std)
        mlflow.log_metric("scaled_mean", scaled_mean)
        mlflow.log_metric("scaled_std", scaled_std)
        mlflow.log_metric("total_samples", len(df_windowed))
        mlflow.log_metric("numeric_features", len(numeric_cols))
        mlflow.log_param("scaler_type", "RobustScaler")
        
        elapsed = time.time() - start_time
        mlflow.log_metric("execution_time_seconds", elapsed)
        
        print(f"\n✓ BAŞARILI! ({elapsed:.2f}s)")
        
    except Exception as e:
        print(f"\n✗ HATA: {e}")
        mlflow.log_param("error", str(e))
        import traceback
        traceback.print_exc()