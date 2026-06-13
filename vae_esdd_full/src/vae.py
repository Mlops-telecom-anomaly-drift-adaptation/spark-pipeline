# src/vae.py
"""
VAE Mimarisi — Makalenin temel bileşeni (Bölüm 4.1)

Autoencoder yapısı:
    Encoder : x  →  μ, log σ²        (gizli dağılım parametreleri)
    Sampler : z  =  μ + σ·ε          (reparameterization trick)
    Decoder : z  →  x̂                (rekonstrüksiyon)

Kayıp (Eq.2):
    l_VAE = l_AE(x, x̂) + β · l_KL(x)
    l_KL  = 0.5 · Σ(μ² + σ² − log σ² − 1)   [Eq.1]

Anomali skoru = l_VAE(x, x̂)
Eşik (Eq.3)  = mean(L) + std(L)       ← adaptif
Tahmin (Eq.4) = 1 if l_VAE > eşik
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dims, latent_dim,
                 beta=1.0, lr=0.001, seed=42):
        """
        Parametreler (Tablo 3 — Makale):
            input_dim   : özellik sayısı
            hidden_dims : gizli katman boyutları, örn. [64, 8]
            latent_dim  : latent uzay boyutu, örn. 2
            beta        : KL ağırlığı (varsayılan 1.0)
            lr          : öğrenme hızı (varsayılan 0.001)
        """
        super().__init__()
        torch.manual_seed(seed)
        self.latent_dim = latent_dim
        self.beta = beta

        # ── ENCODER ──────────────────────────────────────────────────
        # x → gizli katmanlar → (μ, log σ²)
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.LeakyReLU(0.1)]
            prev = h
        self.encoder_body = nn.Sequential(*enc)
        self.fc_mu         = nn.Linear(prev, latent_dim)   # ortalama μ
        self.fc_logvar     = nn.Linear(prev, latent_dim)   # log σ²

        # ── DECODER ──────────────────────────────────────────────────
        # z → ters gizli katmanlar → x̂
        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.LeakyReLU(0.1)]
            prev = h
        dec += [nn.Linear(prev, input_dim), nn.Sigmoid()]
        self.decoder_body = nn.Sequential(*dec)

        # He initialization (Tablo 3)
        self._init_weights()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)

    # ── FORWARD PASS ─────────────────────────────────────────────────
    def encode(self, x):
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Eğitimde örnekle, testte μ kullan"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        return self.decoder_body(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterize(mu, logvar)
        x_hat      = self.decode(z)
        return x_hat, mu, logvar

    # ── KAYIP FONKSİYONU ─────────────────────────────────────────────
    def _loss(self, x, x_hat, mu, logvar):
        """
        l_VAE = l_AE + β·l_KL   [Eq.2]
        """
        # Rekonstrüksiyon: Binary Cross-Entropy (Sea, Circle, Sine, MNIST)
        bce = -(x * torch.log(x_hat + 1e-8) +
                (1 - x) * torch.log(1 - x_hat + 1e-8))
        l_AE = bce.sum(dim=1).mean()

        # KL sapması [Eq.1]
        l_KL = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        return l_AE + self.beta * l_KL

    def _loss_vec(self, x, x_hat, mu, logvar):
        """Her örnek için ayrı kayıp (anomali skoru olarak kullanılır)"""
        bce   = -(x * torch.log(x_hat + 1e-8) +
                  (1-x) * torch.log(1 - x_hat + 1e-8))
        l_AE  = bce.sum(dim=1)
        l_KL  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return l_AE + self.beta * l_KL

    # ── EĞİTİM ───────────────────────────────────────────────────────
    def fit(self, X_np, num_epochs=10, batch_size=64):
        """
        Eq.5: h^(t+Δ) = h^t.train(mov_train(i))
        Incremental güncelleme — yeni veriyle ince ayar
        """
        self.train()
        X_t     = torch.tensor(X_np, dtype=torch.float32)
        dataset = TensorDataset(X_t)
        loader  = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, drop_last=False)
        for _ in range(num_epochs):
            for (batch,) in loader:
                if len(batch) < 2:
                    continue
                self.optimizer.zero_grad()
                x_hat, mu, logvar = self(batch)
                loss = self._loss(batch, x_hat, mu, logvar)
                loss.backward()
                self.optimizer.step()
        self.eval()

    # ── TAHMİN / SKOR ────────────────────────────────────────────────
    def score(self, X_np):
        """Her örnek için anomali skoru döndür (l_VAE)"""
        self.eval()
        with torch.no_grad():
            x     = torch.tensor(np.atleast_2d(X_np), dtype=torch.float32)
            x_hat, mu, logvar = self(x)
            scores = self._loss_vec(x, x_hat, mu, logvar)
        return scores.numpy()
