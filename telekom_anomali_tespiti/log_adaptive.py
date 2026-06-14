import mlflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("telekom-model-karsilastirma")

m = {"precision": 0.7432, "recall": 0.6825, "f1": 0.7076, "auc": 0.6894,
     "drift_alarmi": 272, "test_pencere_sayisi": 42996}

with mlflow.start_run(run_name="VAEplusESDD-Adaptasyon-Acik"):
    mlflow.log_param("model", "VAEplusESDD")
    mlflow.log_param("mod", "adaptasyon_acik_streaming")
    mlflow.log_param("test_pencere_sayisi", m["test_pencere_sayisi"])
    mlflow.log_param("drift_alarmi", m["drift_alarmi"])
    mlflow.log_metric("macro_precision", m["precision"])
    mlflow.log_metric("macro_recall", m["recall"])
    mlflow.log_metric("macro_f1", m["f1"])
    mlflow.log_metric("roc_auc", m["auc"])
    print("loglandi: VAEplusESDD-Adaptasyon-Acik")
print("Bitti.")