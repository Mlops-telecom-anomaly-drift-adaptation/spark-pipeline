import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

exp = client.get_experiment_by_name("telekom-model-karsilastirma")
runs = client.search_runs(exp.experiment_id)

for r in runs:
    ad = r.data.tags.get("mlflow.runName", "")
    if ad == "VAEplusESDD" and r.data.params.get("mod") is None:
        client.log_param(r.info.run_id, "mod", "adaptasyon_kapali")
        print("etiket eklendi:", ad, r.info.run_id)
    if ad == "LSTM-VAE" and r.data.params.get("mod") is None:
        client.log_param(r.info.run_id, "mod", "referans_baseline")
        print("etiket eklendi:", ad, r.info.run_id)

print("Bitti.")