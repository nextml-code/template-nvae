from torch import nn
from workflow.torch import ModuleCompose

from vae.architecture import module


def ResidualConvBlock(external_channels, internal_channels, dilation):
    return module.CenterSliceResidual(nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(external_channels, internal_channels, 3, 1, 0, dilation),
        nn.ReLU(),
        nn.Conv2d(internal_channels, external_channels, 1, 1, 0),
    ))


def ResidualStack(external_channels, internal_channels, depth):
    return ModuleCompose(*[
        ResidualConvBlock(
            external_channels, internal_channels, 3 ** dilation_index
        )
        for dilation_index in range(depth)
    ])
