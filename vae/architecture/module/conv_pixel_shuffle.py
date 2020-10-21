import torch
from torch import nn


class ConvPixelShuffle(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        upscale_factor=2,
    ):
        super().__init__()
        self.convolution = torch.nn.Conv2d(
            in_channels,
            out_channels * upscale_factor ** 2,
            kernel_size=1,
            padding=0,
        )

        self.upsampled = torch.nn.PixelShuffle(upscale_factor)
        self.initializer = torch.nn.init.kaiming_normal_
        self.convolution.weight.data = (
            self.icnr_initialization(self.convolution.weight.data)
        )

    def icnr_initialization(self, tensor):
        if self.upsampled.upscale_factor == 1:
            return self.initializer(tensor)

        new_shape = (
            [int(tensor.shape[0] / (self.upsampled.upscale_factor ** 2))]
            + list(tensor.shape[1:])
        )

        subkernel = self.initializer(torch.zeros(new_shape)).transpose(0, 1)

        kernel = (
            subkernel.reshape(subkernel.shape[0], subkernel.shape[1], -1)
            .repeat(1, 1, self.upsampled.upscale_factor ** 2)
        )

        return (
            kernel.reshape([-1, tensor.shape[0]] + list(tensor.shape[2:]))
            .transpose(0, 1)
        )

    def forward(self, inputs):
        return self.upsampled(self.convolution(inputs))
