from functools import partial
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
from workflow.torch import module_device, ModuleCompose

from vae.problem import settings
from vae.architecture import module


class DecoderCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        expanded_channels = channels * 6
        self.seq = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            module.Swish(),
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=5, padding=2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            module.Swish(),
            nn.Conv2d(expanded_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            module.SqueezeExcitation(channels),
        )

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        # return (x + self.seq(x)).float()
        return x + self.seq(x)


def AbsoluteVariationalBlock(feature_shape, latent_channels):
    channels = feature_shape[1]
    return module.AbsoluteVariationalBlock(
        # feature -> sample
        sample=module.AbsoluteVariational(
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
            nn.Conv2d(latent_channels, channels, kernel_size=1, bias=False),
            DecoderCell(channels),
        ),
        # decoded_sample -> upsample / previous
        computed=ModuleCompose(
            DecoderCell(channels),
        ),
    )


def RelativeVariationalBlock(previous_shape, feature_shape, latent_channels):
    channels = feature_shape[1]
    return module.RelativeVariationalBlock(
        # previous, feature -> sample
        sample=module.RelativeVariational(
            # previous -> absolute_parameters
            absolute_parameters=ModuleCompose(
                # DecoderCell(previous_shape[1]),
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
                # DecoderCell(previous_shape[1] + feature_shape[1]),
                nn.Conv2d(previous_shape[1] + feature_shape[1], channels, kernel_size=1),
                module.Swish(),
                nn.Conv2d(channels, latent_channels * 2, kernel_size=1),
                partial(torch.chunk, chunks=2, dim=1),
            ),
        ),
        # sample -> decoded_sample
        decoded_sample=ModuleCompose(
            module.RandomFourier(8),
            nn.Conv2d(latent_channels + 8, channels, kernel_size=1, bias=False),
            DecoderCell(channels),
        ),
        # decoded_sample, previous -> upsample / previous
        computed=ModuleCompose(
            lambda decoded_sample, previous: (
                torch.cat([decoded_sample, previous], dim=1)
            ),
            DecoderCell(channels + previous_shape[1]),
            nn.Conv2d(
                channels + previous_shape[1],
                channels,
                kernel_size=1,
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
        upsampled_blocks = list()
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
                )
                previous, _ = relative_variational_block(
                    previous, example_feature
                )
                inner_blocks.append(relative_variational_block)

            relative_variational_blocks.append(nn.ModuleList(inner_blocks))
            upsample = module.ConvPixelShuffle(
                previous.shape[1],
                example_feature.shape[1] // 2,
            )
            previous = upsample(previous)
            upsampled_blocks.append(upsample)
            
        self.relative_variational_blocks = nn.ModuleList(
            relative_variational_blocks
        )
        self.upsampled_blocks = nn.ModuleList(upsampled_blocks)

        self.n_mixture_components = 5

        print('previous.shape:', previous.shape)
        self.image = ModuleCompose(
            DecoderCell(previous.shape[1]),
            nn.BatchNorm2d(previous.shape[1]),
            nn.Conv2d(
                previous.shape[1],
                3 * 3 * self.n_mixture_components,
                kernel_size=1,
            ),
            lambda x: x.view(
                -1, 3, 3 * self.n_mixture_components,
                settings.HEIGHT, settings.WIDTH
            ),
            lambda x: x.permute(0, 1, 3, 4, 2),
            lambda x: x.chunk(3, dim=-1),
            lambda logits, unlimited_loc, unlimited_scale: (
                logits,
                torch.tanh(unlimited_loc),
                F.softplus(unlimited_scale),
            ),
        )

    def forward(self, features):
        head, kl = self.absolute_variational_block(features[-1])

        kl_losses = [kl]
        for feature, blocks, upsampled in zip(
            reversed(features),
            self.relative_variational_blocks,
            self.upsampled_blocks,
        ):
            for block in blocks:
                head, relative_kl = block(head, feature)
                kl_losses.append(relative_kl)
            head = upsampled(head)

        return (
            self.image(head),
            kl_losses,
        )

    def generated(self, shape, prior_std):
        head = self.absolute_variational_block.generated(shape, prior_std)

        for blocks, upsampled in zip(
            self.relative_variational_blocks, self.upsampled_blocks
        ):
            for block in blocks:
                head = block.generated(head, prior_std)
            head = upsampled(head)

        return self.image(head)

    def partially_generated(self, features, shape, sample, prior_std):
        if sample[0]:
            head = self.absolute_variational_block.generated(shape, prior_std)
        else:
            head, _ = self.absolute_variational_block(features[-1])

        for feature, blocks, upsampled, inner_sample in zip(
            reversed(features),
            self.relative_variational_blocks,
            self.upsampled_blocks,
            sample,
        ):
            for block in blocks:
                if inner_sample:
                    head = block.generated(head, prior_std)
                else:
                    head, _ = block(head, feature)
            head = upsampled(head)

        return self.image(head)
