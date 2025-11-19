import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder1D(nn.Module):
    def __init__(self, input_len, feature_dim=128):
        super().__init__()
        # CNN Adaptativa
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(
                32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(
                64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(
                128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            # Isso é mágico: funciona para qualquer tamanho de input!
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return self.fc(x)


class LeJEPA(nn.Module):
    def __init__(self, input_len, feature_dim=128, lambda_weight=0.6):
        super().__init__()
        self.encoder = Encoder1D(input_len, feature_dim)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.BatchNorm1d(
                feature_dim), nn.ReLU(),
            nn.Linear(feature_dim, 64)
        )
        self.lambda_w = lambda_weight

        # SIGReg Buffers
        t = torch.linspace(-5, 5, 20)
        self.register_buffer('t', t)
        self.weight = torch.exp(-0.5 * t**2)

    def sigreg_loss(self, z):
        N, D = z.shape
        theta = torch.randn(D, 16, device=z.device)
        theta = theta / (torch.norm(theta, p=2, dim=0, keepdim=True) + 1e-8)
        z_proj = z @ theta

        zp = z_proj.unsqueeze(-1)
        t_grid = self.t.view(1, 1, -1)
        args = zp * t_grid

        ecf_real = torch.cos(args).mean(dim=0)
        ecf_imag = torch.sin(args).mean(dim=0)
        target_real = torch.exp(-0.5 * self.t**2).unsqueeze(0)

        return ((ecf_real - target_real)**2 + ecf_imag**2).mean() * self.weight.mean()

    def forward(self, v1, v2):
        h1 = self.encoder(v1)
        z1 = self.projector(h1)
        h2 = self.encoder(v2)
        z2 = self.projector(h2)

        loss_pred = F.mse_loss(z1, z2)
        loss_sig = (self.sigreg_loss(z1) + self.sigreg_loss(z2)) / 2

        return (1 - self.lambda_w) * loss_pred + self.lambda_w * loss_sig
