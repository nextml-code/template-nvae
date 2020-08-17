import numpy as np
import torch
from torch import nn
from workflow.torch import module_device

from mnist_nvae.architecture import module


class Variational(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.variational_parameters = parameters

    @staticmethod
    def sample(mean, log_variance):
        std = torch.exp(0.5 * log_variance)
        return mean + torch.randn_like(std) * std

    @staticmethod
    def kl(mean, log_variance):
        loss = -0.5 * (1 + log_variance - mean ** 2 - torch.exp(log_variance))
        # return loss.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        return loss.mean()

    def forward(self, feature):
        mean, log_variance = self.variational_parameters(feature)
        return (
            Variational.sample(mean, log_variance),
            Variational.kl(mean, log_variance),
        )

    def generated(self, shape):
        return torch.randn(shape).to(module_device(self))


class RelativeVariational(nn.Module):
    def __init__(self, absolute_parameters, relative_parameters):
        super().__init__()
        self.absolute_parameters = absolute_parameters
        self.relative_parameters = relative_parameters

    @staticmethod
    def kl(mean, log_variance, delta_mean, delta_log_variance):
        var = torch.exp(log_variance)
        delta_var = torch.exp(delta_log_variance)
        loss = -0.5 * (
            1 + delta_log_variance - delta_mean ** 2 / var - delta_var
        )
        # return loss.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        return loss.mean()

    def forward(self, previous, feature):
        mean, log_variance = self.absolute_parameters(previous)
        delta_mean, delta_log_variance = self.relative_parameters(
            previous, feature
        )
        return (
            Variational.sample(
                mean + delta_mean, log_variance + delta_log_variance
            ),
            RelativeVariational.kl(
                mean, log_variance, delta_mean, delta_log_variance
            ),
        )

    def generated(self, previous):
        mean, log_variance = self.absolute_parameters(previous)
        return Variational.sample(mean, log_variance)


class VariationalBlock(nn.Module):
    def __init__(self, sample, decoded_sample, upsample):
        super().__init__()
        self.sample = sample
        self.decoded_sample = decoded_sample
        self.upsample = upsample

    def forward(self, head):
        sample, kl = self.sample(head)
        upsample = self.upsample(
            self.decoded_sample(sample)
        )
        return upsample, kl

    def generated(self, shape):
        return self.upsample(
            self.decoded_sample(
                self.sample.generated(shape)
            )
        )


class RelativeVariationalBlock(nn.Module):
    def __init__(self, sample, decoded_sample, upsample):
        super().__init__()
        self.sample = sample
        self.decoded_sample = decoded_sample
        self.upsample = upsample

    def forward(self, previous, feature):
        sample, kl = self.sample(previous, feature)
        upsample = self.upsample(
            self.decoded_sample(sample),
            previous,
        )
        return upsample, kl

    def generated(self, previous):
        return self.upsample(
            self.decoded_sample(
                self.sample.generated(previous)
            ),
            previous,
        )
