import torch
import torch.nn as nn

class TabularMAE(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=32, latent_dim=16):
        super(TabularMAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, mask_ratio=0.2):
        if self.training and mask_ratio > 0:
            mask = torch.rand_like(x) > mask_ratio
            x_masked = x * mask
        else:
            x_masked = x
        return self.decoder(self.encoder(x_masked))