import mlflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("telekom-model-karsilastirma")

m = {"precision": 0.7487, "recall": 0.7711, "f1": 0.7593, "auc": 0.9162}

with mlflow.start_run(run_name="LSTM-VAE-Adaptif-Tuned"):
    mlflow.log_param("model", "LSTM-VAE")
    mlflow.log_param("mod", "adaptasyon_acik_tuned")
    mlflow.log_param("test_pencere_sayisi", 42996)
    mlflow.log_param("adaptasyon", "drift-tetikli adaptif esik (KS testi)")
    mlflow.log_param("tuning", "Optuna 60 deneme")
    mlflow.log_param("w1_skor_agirligi", 0.8555)
    mlflow.log_param("kalibrasyon_kstd", 1.7872)
    mlflow.log_metric("macro_precision", m["precision"])
    mlflow.log_metric("macro_recall", m["recall"])
    mlflow.log_metric("macro_f1", m["f1"])
    mlflow.log_metric("roc_auc", m["auc"])
    print("loglandi: LSTM-VAE-Adaptif-Tuned")
print("Bitti.")