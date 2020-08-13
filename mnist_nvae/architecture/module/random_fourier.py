import numpy as np
import torch
import torch.nn as nn

from mnist_nvae.architecture import module


class RandomFourier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.B = nn.Parameter(
            torch.randn((in_channels, out_channels // 2)), requires_grad=False
        )

    @staticmethod
    def linspace(x, fourier_length):
        return (
            torch.linspace(0, 1, steps=fourier_length)
            .to(x)
            .view(1, 1, -1)
            .repeat(x.shape[0], 1, x.shape[-1] // fourier_length)
        )

    @staticmethod
    def grid(x, fourier_length):
        pass

    def forward(self, x):
        x_proj = (2 * np.pi * x.transpose(1, -1)) @ self.B
        return (
            torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
            .transpose(1, -1)
        )


class FourierSampleDecoder(nn.Module):
    def __init__(self, fourier_channels, fourier_length, decoded):
        super().__init__()
        self.fourier_length = fourier_length
        self.random_fourier = RandomFourier(1, fourier_channels)
        self.decoded = decoded

    def forward(self, sample):
        return self.decoded(
            torch.cat([
                self.random_fourier(module.RandomFourier.linspace(
                    sample, self.fourier_length
                )),
                sample,
            ], dim=1)
        )
