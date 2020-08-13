import numpy as np
import torch
from torch import nn


# TODO: Momentum. Keep track of weight of current estimates

class NormalStandardizer(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.n_calls = 0
        self.mean = nn.Parameter(torch.zeros(n_channels), requires_grad=False)
        self.std = nn.Parameter(torch.zeros(n_channels), requires_grad=False)

    def expand_dims(self, x, like):
        extra_dims = np.ones(like.ndim, dtype=np.int32)[2:]
        return x.view(1, self.n_channels, *extra_dims)

    def forward(self, x):
        if self.training:
            flat = torch.flatten(x.detach().transpose(0, 1), start_dim=1)
            mean = flat.mean(dim=-1)
            std = flat.std(dim=-1)

            self.mean.data = (self.mean * self.n_calls + mean) / (self.n_calls + 1)
            self.std.data = (self.std * self.n_calls + std) / (self.n_calls + 1)
            # expected standard deviation of a batch rather than the population
            
            self.n_calls += 1

        return (
            (x - self.expand_dims(self.mean, x))
            / (self.expand_dims(self.std, x) + 1e-4)
        )
