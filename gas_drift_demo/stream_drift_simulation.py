import time
import os
import re
import glob

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from river.drift import ADWIN
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = "/Users/zeynep/Desktop/SparkProjem/drift_final/Dataset"

N_FEATURES = 128
SLEEP_TIME = 2

PUSHGATEWAY_ADDRESS = "http://localhost:9091"
JOB_NAME = "dataset_drift_stream"

# RF
N_ESTIMATORS = 200
RANDOM_STATE = 42


# ============================================================
# PROMETHEUS METRICS
# ============================================================
registry = CollectorRegistry()

# GLOBAL STATE (label’sız) -> tek seri
g_current_batch = Gauge(
    "current_batch",
    "Currently processed batch number (global state)",
    registry=registry
)

# BATCH-SCOPED metrics (batch label'lı)
g_batch_samples = Gauge(
    "batch_samples",
    "Number of samples in current batch",
    ["batch"],
    registry=registry
)

g_accuracy = Gauge(
    "model_accuracy",
    "Accuracy (prev->current batch)",
    ["batch"],
    registry=registry
)

g_f1_macro = Gauge(
    "model_f1_macro",
    "F1-macro (prev->current batch)",
    ["batch"],
    registry=registry
)

g_precision_macro = Gauge(
    "model_precision_macro",
    "Precision-macro (prev->current batch)",
    ["batch"],
    registry=registry
)

g_recall_macro = Gauge(
    "model_recall_macro",
    "Recall-macro (prev->current batch)",
    ["batch"],
    registry=registry
)

g_error_rate = Gauge(
    "model_error_rate",
    "Error rate (1-accuracy) (prev->current batch)",
    ["batch"],
    registry=registry
)

g_drift = Gauge(
    "drift_detected",
    "ADWIN drift detected (1/0) for this batch",
    ["batch"],
    registry=registry
)

g_retrain = Gauge(
    "retrain_performed",
    "Model retrained on this batch (1/0)",
    ["batch"],
    registry=registry
)


# ============================================================
# HELPERS
# ============================================================
def natural_batch_key(path: str) -> int:
    """Sort batch10 after batch9 (not between batch1 and batch2)."""
    m = re.search(r"batch(\d+)\.dat$", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def find_batches(folder: str):
    files = glob.glob(os.path.join(folder, "batch*.dat"))
    files = sorted(files, key=natural_batch_key)
    return files


def load_batch(path: str, n_features: int = N_FEATURES):
    if not os.path.exists(path):
        return None, None

    X, y = load_svmlight_file(path, n_features=n_features)
    X = X.toarray().astype(np.float32)
    y = y.astype(np.int32)  # sınıflar 1..6 gibi geliyor
    return X, y


def safe_push():
    push_to_gateway(PUSHGATEWAY_ADDRESS, job=JOB_NAME, registry=registry)


# ============================================================
# MAIN STREAM
# ============================================================
def run_stream():
    print("\n🚀 Stream drift simülasyonu başladı\n")
    print(f"📁 Dataset folder: {DATA_FOLDER}")

    batch_files = find_batches(DATA_FOLDER)
    if not batch_files:
        raise FileNotFoundError(f"batch*.dat bulunamadı: {DATA_FOLDER}")

    print(f"🔍 {len(batch_files)} batch bulundu:")
    for bf in batch_files:
        print(" -", os.path.basename(bf))

    # Model + Drift detector
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    adwin = ADWIN()

    # ---- init: train on batch1
    b1_path = batch_files[0]
    b1_num = natural_batch_key(b1_path)

    X_prev, y_prev = load_batch(b1_path)
    if X_prev is None:
        raise FileNotFoundError(f"Batch okunamadı: {b1_path}")

    model.fit(X_prev, y_prev)

    # Push initial state (batch1 cache/train done)
    g_current_batch.set(b1_num)
    g_batch_samples.labels(batch=str(b1_num)).set(int(X_prev.shape[0]))

    # batch1 için performans metrikleri yok (prev->current yok), NaN basmak yerine basmıyoruz.
    g_drift.labels(batch=str(b1_num)).set(0)
    g_retrain.labels(batch=str(b1_num)).set(0)

    try:
        safe_push()
        print(f"📡 (init) Pushgateway'e gönderildi: batch{b1_num}")
    except Exception as e:
        print(f"❌ Pushgateway hatası (init): {e}")

    time.sleep(SLEEP_TIME)

    # ---- stream: for batch2..N
    for path in batch_files[1:]:
        batch_num = natural_batch_key(path)
        b = str(batch_num)

        print(f"\n⏳ Batch {batch_num} işleniyor...")

        X_cur, y_cur = load_batch(path)
        if X_cur is None:
            print(f"❌ Okunamadı: {path}")
            break

        # global state
        g_current_batch.set(batch_num)
        g_batch_samples.labels(batch=b).set(int(X_cur.shape[0]))

        # eval prev->current
        y_pred = model.predict(X_cur)

        acc = accuracy_score(y_cur, y_pred)
        err = 1.0 - acc
        f1m = f1_score(y_cur, y_pred, average="macro", zero_division=0)
        precm = precision_score(y_cur, y_pred, average="macro", zero_division=0)
        recm = recall_score(y_cur, y_pred, average="macro", zero_division=0)

        # drift detection (error stream)
        drift_flag = 0
        for yt, yp in zip(y_cur, y_pred):
            adwin.update(int(yt != yp))
            if getattr(adwin, "drift_detected", False):
                drift_flag = 1
                break

        retrain_flag = 0
        if drift_flag:
            print("⚠️ DRIFT tespit edildi → retrain (current batch ile)")
            model.fit(X_cur, y_cur)
            adwin = ADWIN()   # reset detector
            retrain_flag = 1
        else:
            print("✅ Stabil (drift yok)")

        # set metrics
        g_accuracy.labels(batch=b).set(float(acc))
        g_error_rate.labels(batch=b).set(float(err))
        g_f1_macro.labels(batch=b).set(float(f1m))
        g_precision_macro.labels(batch=b).set(float(precm))
        g_recall_macro.labels(batch=b).set(float(recm))
        g_drift.labels(batch=b).set(int(drift_flag))
        g_retrain.labels(batch=b).set(int(retrain_flag))

        # push
        try:
            safe_push()
            print(f"📡 Push OK | acc={acc:.4f} f1={f1m:.4f} drift={drift_flag} retrain={retrain_flag}")
        except Exception as e:
            print(f"❌ Pushgateway hatası: {e}")

        # shift window (prev = current)
        X_prev, y_prev = X_cur, y_cur

        time.sleep(SLEEP_TIME)

    print("\n🏁 Stream simülasyonu tamamlandı")


if __name__ == "__main__":
    run_stream()