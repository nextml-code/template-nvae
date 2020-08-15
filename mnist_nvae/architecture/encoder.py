import torch
from torch import nn
from workflow.torch import ModuleCompose

from mnist_nvae.architecture import module


class EncoderCell(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(dim),
            module.Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            module.Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            module.SqueezeExcitation(dim),
        )

    def forward(self, x):
        return x + self.seq(x)


class Encoder(nn.Module):
    def __init__(self, channels, levels):
        super().__init__()
        self.levels = nn.ModuleList([
            ModuleCompose(
                nn.Conv2d(
                    3 if level == 0 else channels * 2 ** (level - 1),
                    channels * 2 ** level, 3,
                    stride=2,
                    padding=1
                ),
                EncoderCell(channels * 2 ** level),
            )
            for level in range(levels)
        ])

    def forward(self, head):
        head = head / 255 * 2 - 1
        features = list()
        for level in self.levels:
            head = level(head)
            features.append(head)
        return features
