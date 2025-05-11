import torch
import torch.nn as nn
import torch.nn.functional as F

class CouplingLayer(nn.Module):
    """RealNVP耦合层，简化版"""
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.scale_net = nn.Sequential(
            nn.Linear(dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim//2),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim//2)
        )
    
    def forward(self, x, invert=False):
        x1, x2 = x.chunk(2, dim=1)
        if not invert:
            s = self.scale_net(x1)
            t = self.translate_net(x1)
            y2 = x2 * torch.exp(s) + t
            return torch.cat([x1, y2], dim=1), s.sum(dim=1)
        else:
            s = self.scale_net(x1)
            t = self.translate_net(x1)
            x2 = (x2 - t) * torch.exp(-s)
            return torch.cat([x1, x2], dim=1)

class SimpleRealNVP(nn.Module):
    """轻量级RealNVP，仅含2个耦合层"""
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(input_dim),
            CouplingLayer(input_dim)
        ])
        self.base_dist = torch.distributions.Normal(0, 1)
        
    def forward(self, x):
        log_det = 0
        for layer in self.layers:
            x, ld = layer(x)
            log_det += ld
        return x, log_det
    
    def invert(self, z):
        for layer in reversed(self.layers):
            z = layer(z, invert=True)
        return z