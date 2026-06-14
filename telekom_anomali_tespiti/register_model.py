import mlflow, mlflow.pyfunc, torch, numpy as np, pandas as pd

class AnomalyModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.cfg = torch.load(context.artifacts["config"], map_location="cpu", weights_only=False)
        self.model = torch.jit.load(context.artifacts["model"], map_location="cpu").eval()
        self.fcols = self.cfg["feature_cols"]; self.ws = self.cfg["window_size"]
        self.scaler = self.cfg["scaler"]; self.thr = float(self.cfg["threshold"])
        self.w1, self.w2 = self.cfg.get("score_weights", (0.3, 0.7))

    def predict(self, context, model_input):
        n = len(self.fcols)
        X = model_input.to_numpy(dtype=np.float32) if hasattr(model_input,"to_numpy") else np.asarray(model_input,np.float32)
        Xs = self.scaler.transform(X).astype(np.float32).reshape(-1, self.ws, n)
        with torch.no_grad():
            xb = torch.tensor(Xs)
            r1, r2, _, _ = self.model(xb)
            e1 = torch.mean((r1-xb)**2, dim=(1,2)); e2 = torch.mean((r2-xb)**2, dim=(1,2))
            scores = (self.w1*e1 + self.w2*e2).numpy()
        preds = (scores >= self.thr).astype(int)
        return pd.DataFrame({"anomaly_score": scores, "prediction": preds,
                             "label_text": np.where(preds==1,"Anomaly","Normal")})

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("anomaly-detection")
with mlflow.start_run(run_name="register-lstm-vae") as run:
    mlflow.log_params({"window_size":24,"threshold":0.020946906879544258,"n_features":33,"score_weights":"0.3,0.7"})
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=AnomalyModel(),
        artifacts={"model":"anomaly_model.pt","config":"anomaly_model_config.pth"},
        registered_model_name="anomaly-detector",
    )
    print("Kaydedildi. run_id:", run.info.run_id)