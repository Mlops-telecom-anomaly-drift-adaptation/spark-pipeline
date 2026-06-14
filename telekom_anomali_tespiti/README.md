\# Telekom Anomali Tespiti — MLflow Pipeline \& Model Karşılaştırması



Bu klasör, gerçek telekom hücresel performans verisi üzerinde anomali tespiti

çalışmasını içerir. İki model MLflow üzerinde kaydedilip karşılaştırılmıştır:

referans LSTM-VAE ile geliştirdiğimiz VAE++ESDD modeli.



\## Pipeline Akışı

1\. `register\_model.py` — Eğitilmiş modeli MLflow Registry'ye kaydeder (pyfunc wrapper).

2\. `infer\_client.py` — Modeli MLflow REST endpoint'inden (mlflow models serve) çağırıp

&#x20;  normal/anormal tahmini alır.

3\. `drift\_detect.py` — Eğitim verisini referans alıp test verisiyle KS testi ile

&#x20;  data drift tespiti yapar.

4\. `adapt.py` — Drift sonrası eşik (threshold) adaptasyonu uygular ve etkisini ölçer.

5\. `log\_compare.py` — İki modelin sonuçlarını MLflow'a karşılaştırma run'ları olarak loglar.

6\. `log\_adaptive.py` — VAE++ESDD'nin adaptasyon-açık (streaming) sonucunu loglar.

7\. `fix\_label.py` — MLflow run'larına açıklayıcı `mod` etiketi ekler.



\## Sonuçlar (telekom test seti, 42.996 pencere)

| Model | Mod | Macro F1 | ROC-AUC |

|-------|-----|----------|---------|

| LSTM-VAE | referans (baseline) | 0.751 | 0.916 |

| VAE++ESDD | adaptasyon kapalı | 0.572 | 0.767 |

| VAE++ESDD | adaptasyon açık | 0.708 | 0.689 |



Not: Veri seti (CSV), eğitilmiş model dosyası (.pt) ve mlruns/ klasörü boyut

nedeniyle repoya dahil edilmemiştir.



\## Çalıştırma

```

pip install mlflow torch scikit-learn pandas numpy scipy

python register\_model.py

mlflow models serve -m "models:/anomaly-detector/1" --port 5001 --env-manager local

python infer\_client.py

python drift\_detect.py

python adapt.py

```

