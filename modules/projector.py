import torch
import torch.nn as nn


class STOPPolicy(nn.Module):
    def __init__(self, feature_size, intermediate_size=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, 2)
        )

    def forward(self, x, temperature=1.0):
        return self.fc(x)

class LatentPolicy(nn.Module):
    def __init__(self, feature_size, intermediate_size=512, deterministic=False):
        super().__init__()
        self.deterministic = deterministic
        self.fc = nn.Sequential(
            nn.Linear(feature_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
        )

        self.mean = nn.Linear(intermediate_size, feature_size)
        if not deterministic:
            self.log_var = nn.Linear(intermediate_size, feature_size)

    def forward(self, x, temperature=1.0):
        x = self.fc(x)
        mean = self.mean(x)
        if self.deterministic:
            return torch.distributions.Normal(mean, torch.ones_like(mean) * 1e-9)
        log_var = self.log_var(x)
        return mean, log_var
