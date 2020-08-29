import numpy as np
import torch
import torch.nn as nn


class RandomFourier(nn.Module):
    def __init__(self, fourier_channels):
        super().__init__()
        if fourier_channels % 2 != 0:
            raise ValueError('Out channel must be divisible by 4')

        self.register_buffer(
            'random_matrix', torch.randn((2, fourier_channels // 2))
        )

    @staticmethod
    def gridspace(x):
        h, w = x.shape[-2:]
        grid_y, grid_x = torch.meshgrid([
            torch.linspace(0, 1, steps=h),
            torch.linspace(0, 1, steps=w)
        ])
        return (
            torch.stack([grid_y, grid_x])
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
            .to(x)
        )

    def forward(self, x):
        gridspace = RandomFourier.gridspace(x)
        projection = (
            (2 * np.pi * gridspace.transpose(1, -1)) @ self.random_matrix
        ).transpose(1, -1)
        return torch.cat([
            x,
            torch.sin(projection),
            torch.cos(projection),
        ], dim=1)
