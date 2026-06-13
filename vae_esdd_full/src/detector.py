# src/detector.py
"""
Drift Dedektörü — Mann-Whitney U Testi (Makale Bölüm 4.2)

Her dedektör iki pencere tutar:
    ref_driftl : başlangıç kayıp referans penceresi (sabit)
    mov_driftl : son W_drift kaybını saklayan kayan pencere

Test (Eq.7–10):
    U hesapla → Z skoru → P değeri
    P < P_warn  → flag_warn  (potansiyel drift)
    P < P_alarm → flag_alarm (gerçek drift alarmı)
"""

import numpy as np
from collections import deque
from scipy.stats import norm


class DriftDetector:

    def __init__(self, w_drift=200, p_warn=0.01, p_alarm=0.001):
        """
        w_drift : pencere boyutu W_drift(i) — makale: random(180,220)
        p_warn  : uyarı eşiği  (varsayılan 0.01)
        p_alarm : alarm eşiği  (varsayılan 0.001)
        """
        self.w_drift    = w_drift
        self.p_warn     = p_warn
        self.p_alarm    = p_alarm
        self.ref_driftl = None                    # referans penceresi
        self.mov_driftl = deque(maxlen=w_drift)   # kayan pencere
        self.flag_warn  = False
        self.flag_alarm = False

    def set_reference(self, losses):
        """İlk eğitim sonrası çağrılır — referans penceresini doldur"""
        self.ref_driftl = np.array(losses[-self.w_drift:], dtype=float)

    def update(self, loss_val):
        """
        Yeni kayıp ekle ve drift testi yap.

        Döndürür:
            'alarm'  → drift var, modeli sıfırla
            'warn'   → potansiyel drift, buffer doldur
            'normal' → drift yok
        """
        self.mov_driftl.append(float(loss_val))
        self.flag_warn  = False
        self.flag_alarm = False

        # Yeterli veri yoksa test yapma
        if self.ref_driftl is None or len(self.mov_driftl) < self.w_drift:
            return 'normal'

        ref = np.array(self.ref_driftl)
        mov = np.array(self.mov_driftl)
        p   = self._mann_whitney_p(ref, mov)

        # Eq.10
        if p < self.p_alarm:
            self.flag_alarm = True
            return 'alarm'
        elif p < self.p_warn:
            self.flag_warn = True
            return 'warn'
        return 'normal'

    def reset(self, new_losses=None):
        """Drift sonrası tüm pencereleri sıfırla"""
        self.mov_driftl.clear()
        self.flag_warn  = False
        self.flag_alarm = False
        if new_losses is not None:
            self.ref_driftl = np.array(new_losses[-self.w_drift:], dtype=float)

    @staticmethod
    def _mann_whitney_p(ref, mov):
        """
        İki pencere arasında Mann-Whitney U testi (Eq.7–9)
        Normal yaklaşım kullanılır (büyük örneklem için geçerli)
        """
        n_r = len(ref)
        n_m = len(mov)

        # Tüm değerleri birleştir ve rank ata
        combined = np.concatenate([ref, mov])
        ranks    = _rank_with_ties(combined)

        R_ref = ranks[:n_r].sum()
        R_mov = ranks[n_r:].sum()

        # Eq.7
        U_ref = n_r * n_m + n_r * (n_r + 1) / 2 - R_ref
        U_mov = n_r * n_m + n_m * (n_m + 1) / 2 - R_mov
        U     = min(U_ref, U_mov)

        # Eq.8 — Normal yaklaşım
        mean_U = n_r * n_m / 2
        std_U  = np.sqrt(n_r * n_m * (n_r + n_m + 1) / 12)
        if std_U == 0:
            return 1.0

        Z = (U - mean_U) / std_U

        # Eq.9 — İki kuyruklu p değeri
        p = 2 * float(norm.cdf(Z))
        return p


def _rank_with_ties(arr):
    """Bağlı değerlere ortalama rank ata"""
    n    = len(arr)
    idx  = np.argsort(arr, kind='stable')
    rank = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and arr[idx[j]] == arr[idx[j + 1]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        rank[idx[i:j + 1]] = avg
        i = j + 1
    return rank
