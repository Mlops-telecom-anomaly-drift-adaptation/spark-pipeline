import mlflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("telekom-model-karsilastirma")

sonuclar = {
    "LSTM-VAE":    {"precision": 0.7309, "recall": 0.7761, "f1": 0.7511, "auc": 0.9159},
    "VAEplusESDD": {"precision": 0.5632, "recall": 0.5870, "f1": 0.5718, "auc": 0.7674},
}
for ad, m in sonuclar.items():
    with mlflow.start_run(run_name=ad):
        mlflow.log_param("model", ad)
        mlflow.log_param("test_pencere_sayisi", 42996)
        mlflow.log_metric("macro_precision", m["precision"])
        mlflow.log_metric("macro_recall", m["recall"])
        mlflow.log_metric("macro_f1", m["f1"])
        mlflow.log_metric("roc_auc", m["auc"])
        print("loglandi:", ad)
print("Bitti.")