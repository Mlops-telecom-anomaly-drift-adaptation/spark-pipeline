# src/model.py
"""
VAE++ESDD Ana Sistemi — Algorithm 1 (Makale Bölüm 4)

İki seviyeli ensemble:
    Level-1 : n VAE → anomali tahmini (P_thre oylama)
    Level-2 : n DriftDetector → drift kararı (D_thre oylama)

Sliding windows:
    mov_train(i) : son W_train(i) örneği saklar (incremental learning)
    ref_driftl(i): başlangıç kayıp referansı (drift tespiti)
    mov_driftl(i): son W_drift(i) kaybı saklar (drift tespiti)
    mov_ESwarn   : uyarı sonrası örnekler (model sıfırlama için)
"""

import numpy as np
from collections import deque
from src.vae import VAE
from src.detector import DriftDetector


class AdaptiveThreshold:
    """Eq.3: θ = mean(L) + std(L)"""
    def __init__(self):
        self.theta = None

    def update(self, losses):
        self.theta = float(np.mean(losses) + np.std(losses))

    def predict(self, loss_val):
        """Eq.4: ŷ = 1 if l_VAE > θ"""
        if self.theta is None:
            return 0
        return int(loss_val > self.theta)


class VAEplusESDD:
    """
    VAE++ESDD — Makale Algorithm 1 tam uygulaması.

    Kullanım:
        model = VAEplusESDD(input_dim=2)
        model.initialize(X_init)            # ilk eğitim
        pred, score, alarm = model.process(x)  # stream'den örnek işle
    """

    def __init__(
        self,
        input_dim,
        hidden_dims=(64, 8),    # Tablo 3: Sea → [64, 8]
        latent_dim=2,
        n=10,                   # ensemble boyutu
        W_train=3000,           # mov_train pencere boyutu
        gamma=2000,             # W_train(i) çeşitliliği: (W_train-gamma, W_train)
        W_drift_min=180,        # W_drift(i) aralığı
        W_drift_max=220,
        P_thre=1,               # anomali karar eşiği
        D_thre=10,              # drift karar eşiği
        P_warn=0.01,
        P_alarm=0.001,
        expiry_time=100,
        beta=1.0,
        lr=0.001,
        num_epochs=10,
        batch_size=64,
        seed=42,
    ):
        self.input_dim   = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim  = latent_dim
        self.n           = n
        self.W_train     = W_train
        self.P_thre      = P_thre
        self.D_thre      = D_thre
        self.expiry_time = expiry_time
        self.beta        = beta
        self.lr          = lr
        self.num_epochs  = num_epochs
        self.batch_size  = batch_size
        self.seed        = seed

        rng = np.random.RandomState(seed)

        # Her üye için farklı pencere boyutu → ensemble çeşitliliği
        self.W_train_i = [rng.randint(W_train - gamma, W_train)
                          for _ in range(n)]
        self.W_drift_i = [rng.randint(W_drift_min, W_drift_max + 1)
                          for _ in range(n)]

        # Sliding windows
        self.mov_train_i = [deque(maxlen=W_train) for _ in range(n)]

        # Level-1: VAE ensemble
        self.vaes       = [self._new_vae() for _ in range(n)]
        self.thresholds = [AdaptiveThreshold() for _ in range(n)]

        # Level-2: Drift dedektör ensemble
        self.detectors = [
            DriftDetector(self.W_drift_i[i], P_warn, P_alarm)
            for i in range(n)
        ]

        # Ensemble buffer (Algorithm 1 satır 17–20)
        self.mov_ESwarn    = deque(maxlen=max(self.W_drift_i))
        self.ES_warn_trig  = 0
        self.ES_alarm_trig = 0
        self.t             = 0

        # Loglama
        self.drift_alarms = []
        self.drift_warns  = []
        self.retrain_log  = []

    # ── İLK EĞİTİM ───────────────────────────────────────────────────
    def initialize(self, X_init):
        """
        Tüm VAE'leri etiket gerektirmeden başlangıç verisiyle eğit.
        Referans pencerelerini ve adaptif eşiği kur.
        """
        print(f"\n[Başlangıç Eğitimi] {len(X_init)} örnek, {self.n} VAE...")
        for i in range(self.n):
            for x in X_init:
                self.mov_train_i[i].append(x)
            self.vaes[i].fit(X_init, self.num_epochs, self.batch_size)
            losses = self.vaes[i].score(X_init)
            self.thresholds[i].update(losses)
            self.detectors[i].set_reference(losses)
        print(f"[Başlangıç Eğitimi] Tamamlandı. "
              f"Örnek eşik: {self.thresholds[0].theta:.4f}")

    # ── ANA DÖNGÜ: tek örnek işle ────────────────────────────────────
    def process(self, x):
        """
        Algorithm 1 — her zaman adımında tek örnek işle.

        Döndürür:
            final_pred  (0/1)  : anomali tahmini
            final_score (float): ortalama anomali skoru
            drift_alarm (bool) : drift tespit edildi mi
        """
        self.t += 1
        preds   = []
        scores  = []
        f_warns = []
        f_alarms = []

        for i in range(self.n):
            # Satır 6: örneği mov_train(i)'a ekle
            self.mov_train_i[i].append(x)

            # Satır 9: anomali skoru ve tahmin
            s = float(self.vaes[i].score(x.reshape(1, -1))[0])
            scores.append(s)
            preds.append(self.thresholds[i].predict(s))

            # Satır 10: drift testi
            result = self.detectors[i].update(s)
            f_warns.append(result in ('warn', 'alarm'))
            f_alarms.append(result == 'alarm')

            # Satır 15–16: scheduled incremental learning
            if self.t % self.W_train_i[i] == 0:
                data = np.array(self.mov_train_i[i])
                self.vaes[i].fit(data, self.num_epochs, self.batch_size)
                losses = self.vaes[i].score(data)
                self.thresholds[i].update(losses)
                self.retrain_log.append(self.t)

        # Satır 11–12: ensemble uyarı tetikleyicisi
        if any(f_warns) and self.ES_warn_trig == 0:
            self.ES_warn_trig = self.t
            self.drift_warns.append(self.t)

        # Satır 17–18: uyarı buffer dolduruluyor
        if self.ES_warn_trig > 0 and self.ES_alarm_trig == 0:
            self.mov_ESwarn.append(x)
            # Satır 19–20: expiry — yanlış uyarıyı iptal et
            if self.t - self.ES_warn_trig > self.expiry_time:
                self.ES_warn_trig = 0
                self.mov_ESwarn.clear()

        # Satır 21–22: drift alarm oylama (D_thre)
        drift_alarm = (sum(f_alarms) >= self.D_thre)
        if drift_alarm:
            self._reset_after_drift()

        # Satır 20: anomali oylama (P_thre = 1)
        final_pred  = int(sum(preds) >= self.P_thre)
        final_score = float(np.mean(scores))
        return final_pred, final_score, drift_alarm

    # ── DRIFT SONRASI SIFIRLAMA ───────────────────────────────────────
    def _reset_after_drift(self):
        """
        Algorithm 1 satır 22:
        Tüm VAE'leri ve pencereleri sıfırla,
        mov_ESwarn verisiyle yeniden eğit.
        """
        self.ES_alarm_trig = self.t
        self.drift_alarms.append(self.t)
        retrain_data = (np.array(self.mov_ESwarn)
                        if len(self.mov_ESwarn) >= self.batch_size
                        else None)

        print(f"  [t={self.t:>6}] DRIFT #{len(self.drift_alarms)} "
              f"— {len(self.mov_ESwarn)} örneğiyle yeniden eğitim...")

        for i in range(self.n):
            self.vaes[i]       = self._new_vae()
            self.thresholds[i] = AdaptiveThreshold()

            if retrain_data is not None:
                self.vaes[i].fit(retrain_data, self.num_epochs, self.batch_size)
                losses = self.vaes[i].score(retrain_data)
                self.thresholds[i].update(losses)
                self.detectors[i].reset(losses)
            else:
                self.detectors[i].reset()

            self.mov_train_i[i].clear()

        self.mov_ESwarn.clear()
        self.ES_warn_trig  = 0
        self.ES_alarm_trig = 0

    def _new_vae(self):
        return VAE(
            input_dim   = self.input_dim,
            hidden_dims = self.hidden_dims,
            latent_dim  = self.latent_dim,
            beta        = self.beta,
            lr          = self.lr,
            seed        = self.seed,
        )
