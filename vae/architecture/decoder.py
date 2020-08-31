from functools import partial
import numpy as np
import torch.nn as nn
import torch
from torch.nn.utils import weight_norm
from workflow.torch import module_device, ModuleCompose

from vae.architecture import module


class DecoderCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        expanded_channels = channels * 6
        self.seq = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, expanded_channels, kernel_size=1),
            nn.BatchNorm2d(expanded_channels),
            module.Swish(),
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=5, padding=2, groups=expanded_channels),
            nn.BatchNorm2d(expanded_channels),
            module.Swish(),
            nn.Conv2d(expanded_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            module.SqueezeExcitation(channels),
        )

    def forward(self, x):
        return x + self.seq(x)


def AbsoluteVariationalBlock(feature_shape, latent_channels):
    channels = feature_shape[1]
    return module.VariationalBlock(
        # feature -> sample
        sample=module.Variational(
            # feature -> absolute_parameters
            parameters=ModuleCompose(
                nn.Conv2d(channels, channels, kernel_size=1),
                module.Swish(),
                nn.Conv2d(channels, latent_channels * 2, kernel_size=1),
                partial(torch.chunk, chunks=2, dim=1),
            )
        ),
        # sample -> decoded_sample
        decoded_sample=ModuleCompose(
            nn.Conv2d(latent_channels, channels, kernel_size=1),
            DecoderCell(channels),
        ),
        # decoded_sample -> upsample / previous
        upsample=ModuleCompose(
            DecoderCell(channels),
            # module.ConvPixelShuffle(
            #     channels,
            #     channels // 2,
            # ),
        ),
    )


def RelativeVariationalBlock(previous_shape, feature_shape, latent_channels, upsample=True):
    channels = feature_shape[1]
    return module.RelativeVariationalBlock(
        # previous, feature -> sample
        sample=module.RelativeVariational(
            # previous -> absolute_parameters
            absolute_parameters=ModuleCompose(
                DecoderCell(previous_shape[1]),
                nn.Conv2d(previous_shape[1], channels, kernel_size=1),
                module.Swish(),
                nn.Conv2d(channels, latent_channels * 2, kernel_size=1),
                partial(torch.chunk, chunks=2, dim=1),
            ),
            # previous, feature -> relative_parameters
            relative_parameters=ModuleCompose(
                lambda previous, feature: (
                    torch.cat([previous, feature], dim=1)
                ),
                DecoderCell(previous_shape[1] + feature_shape[1]),
                nn.Conv2d(previous_shape[1] + feature_shape[1], channels, kernel_size=1),
                module.Swish(),
                nn.Conv2d(channels, latent_channels * 2, kernel_size=1),
                partial(torch.chunk, chunks=2, dim=1),
            ),
        ),
        # sample -> decoded_sample
        decoded_sample=ModuleCompose(
            module.RandomFourier(8),
            nn.Conv2d(latent_channels + 8, channels, kernel_size=1),
            DecoderCell(channels),
        ),
        # decoded_sample, previous -> upsample / previous
        upsample=ModuleCompose(
            lambda decoded_sample, previous: (
                torch.cat([decoded_sample, previous], dim=1)
            ),
            DecoderCell(channels + previous_shape[1]),
            (
                module.ConvPixelShuffle(
                    channels + previous_shape[1],
                    channels // 2,
                )
                if upsample
                else nn.Conv2d(
                    channels + previous_shape[1],
                    channels,
                    kernel_size=1,
                )
            ),
        ),
    )


class DecoderNVAE(nn.Module):
    def __init__(self, example_features, latent_channels, level_sizes):
        super().__init__()
        print('example_feature.shape:', example_features[-1].shape)
        self.absolute_variational_block = AbsoluteVariationalBlock(
            example_features[-1].shape, latent_channels
        )
        previous, _ = self.absolute_variational_block(example_features[-1])
        self.latent_height = example_features[-1].shape[-2]
        self.latent_width = example_features[-1].shape[-1]

        relative_variational_blocks = list()
        for level_index, (level_size, example_feature) in enumerate(zip(
            level_sizes, reversed(example_features)
        )):
            print('level_index:', level_index)
            inner_blocks = list()
            for block_index in range(
                1 if level_index == 0 else 0, level_size
            ):
                print('block_index:', block_index)
                print('previous.shape:', previous.shape)
                print('example_feature.shape:', example_feature.shape)
                
                relative_variational_block = RelativeVariationalBlock(
                    previous.shape,
                    example_feature.shape,
                    latent_channels,
                    upsample=(block_index == (level_size - 1)),
                )
                previous, _ = relative_variational_block(
                    previous, example_feature
                )
                inner_blocks.append(relative_variational_block)
            relative_variational_blocks.append(nn.ModuleList(inner_blocks))

        self.relative_variational_blocks = nn.ModuleList(
            relative_variational_blocks
        )

        print('previous.shape:', previous.shape)
        self.image = ModuleCompose(
            DecoderCell(previous.shape[1]),
            nn.BatchNorm2d(previous.shape[1]),
            nn.Conv2d(previous.shape[1], 3, kernel_size=1),
            torch.sigmoid,
            lambda x: x * 255,
        )

    def forward(self, features):
        head, kl = self.absolute_variational_block(features[-1])

        kl_losses = [kl]
        for feature, blocks in zip(
            reversed(features), self.relative_variational_blocks
        ):
            for block in blocks:
                head, relative_kl = block(head, feature)
                kl_losses.append(relative_kl)

        return (
            self.image(head),
            kl_losses,
        )

    def generated(self, shape):
        head = self.absolute_variational_block.generated(shape)

        for blocks in self.relative_variational_blocks:
            for block in blocks:
                head = block.generated(head)

        return self.image(head)

    def partially_generated(self, features, shape, sample):
        if sample[0]:
            head = self.absolute_variational_block.generated(shape)
        else:
            head, _ = self.absolute_variational_block(features[-1])

        for feature, blocks, inner_sample in zip(
            reversed(features), self.relative_variational_blocks, sample
        ):
            for block in blocks:
                if inner_sample:
                    head = block.generated(head)
                else:
                    head, _ = block(head, feature)

        return self.image(head)
