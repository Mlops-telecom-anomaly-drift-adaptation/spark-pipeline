# src/evaluation.py
"""
Değerlendirme Metrikleri — Makale Bölüm 5.2

Prequential (fading factor=0.99):
    Recall      = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    G-mean      = sqrt(Recall × Specificity)

PAUC (Prequential AUC, sliding window d=1000):
    Eq.15: her pencerede AUC hesapla
"""

import numpy as np
from collections import deque
from sklearn.metrics import roc_auc_score


class PrequentialEvaluator:

    def __init__(self, fading=0.99, pauc_window=1000):
        """
        fading      : fading factor (varsayılan 0.99 — Bölüm 5.2)
        pauc_window : PAUC sliding window boyutu (varsayılan 1000)
        """
        self.f = fading
        # Fading counter'lar
        self.TP = self.TN = self.FP = self.FN = 0.0

        # PAUC için sliding window
        self.score_win = deque(maxlen=pauc_window)
        self.label_win = deque(maxlen=pauc_window)

        # Zaman serisi geçmişi
        self.gmean_hist = []
        self.recall_hist = []
        self.spec_hist   = []
        self.pauc_hist   = []

    def update(self, y_true, y_pred, score):
        """Her zaman adımında bir örnek ekle"""
        f = self.f
        if y_true == 1:
            self.TP = f * self.TP + (1 if y_pred == 1 else 0)
            self.FN = f * self.FN + (0 if y_pred == 1 else 1)
        else:
            self.TN = f * self.TN + (1 if y_pred == 0 else 0)
            self.FP = f * self.FP + (0 if y_pred == 0 else 1)

        recall = self.TP / (self.TP + self.FN + 1e-9)
        spec   = self.TN / (self.TN + self.FP + 1e-9)
        gmean  = np.sqrt(recall * spec)

        self.recall_hist.append(recall)
        self.spec_hist.append(spec)
        self.gmean_hist.append(gmean)

        # PAUC
        self.score_win.append(score)
        self.label_win.append(y_true)
        labels = np.array(self.label_win)
        scores = np.array(self.score_win)
        if len(np.unique(labels)) == 2:
            try:
                pauc = roc_auc_score(labels, scores)
            except Exception:
                pauc = 0.5
        else:
            pauc = 0.5
        self.pauc_hist.append(pauc)

    def summary(self):
        """Ortalama metrikler"""
        return {
            'G-mean'     : round(float(np.mean(self.gmean_hist)),  4),
            'Recall'     : round(float(np.mean(self.recall_hist)), 4),
            'Specificity': round(float(np.mean(self.spec_hist)),   4),
            'PAUC'       : round(float(np.mean(self.pauc_hist)),   4),
        }

    def print_summary(self, drift_times=None, drift_alarms=None):
        m = self.summary()
        print("\n" + "="*55)
        print("  SONUÇLAR")
        print("="*55)
        for k, v in m.items():
            print(f"  {k:<16}: {v:.4f}")
        if drift_times:
            print(f"  Gerçek drift   : {drift_times}")
        if drift_alarms:
            print(f"  Tespit edilen  : {drift_alarms}")
        print("="*55)
        return m
