from skmultiflow.drift_detection import ADWIN
import numpy as np

adwin = ADWIN()
data = np.concatenate([np.random.normal(0, 1, 500),
                        np.random.normal(3, 1, 500)])

for i, x in enumerate(data):
    adwin.add_element(x)
    if adwin.detected_change():
        print(f"Drift detected at index {i}")
