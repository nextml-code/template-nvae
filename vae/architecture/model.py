from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from workflow.torch import ModuleCompose, module_device

from vae import architecture, problem


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.level_sizes = [2, 2, 2, 4, 8]
        self.levels = len(self.level_sizes)
        self.encoder = architecture.Encoder(32, levels=self.levels)
        
        self.latent_channels = 20
        self.decoder = architecture.DecoderNVAE(
            example_features=self.encoder(torch.zeros(
                1, 3, problem.settings.HEIGHT, problem.settings.WIDTH
            )),
            latent_channels=self.latent_channels,
            level_sizes=self.level_sizes,
        )

        for module in self.modules():
            if (
                isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
                and len(module._forward_pre_hooks) == 0
            ):
                torch.nn.utils.spectral_norm(module)

    def forward(self, image_batch):
        image_batch = image_batch.to(module_device(self))
        features = self.encoder(image_batch)
        predicted_image, kl_losses = self.decoder(features)
        return architecture.PredictionBatch(
            predicted_image=predicted_image,
            kl_losses=kl_losses,
        )

    def prediction(self, features_batch: architecture.FeaturesBatch):
        return self(features_batch.image_batch)

    def generated(self, n_samples, prior_std):
        predicted_image = self.decoder.generated(
            (
                n_samples,
                self.latent_channels,
                self.decoder.latent_height,
                self.decoder.latent_width,
            ),
            prior_std,
        )
        return architecture.PredictionBatch(predicted_image=predicted_image)

    def partially_generated(self, image_batch, sample, prior_std):
        image_batch = image_batch.to(module_device(self))
        predicted_image = self.decoder.partially_generated(
            self.encoder(image_batch),
            (
                len(image_batch),
                self.latent_channels,
                self.decoder.latent_height,
                self.decoder.latent_width,
            ),
            sample,
            prior_std,
        )
        return architecture.PredictionBatch(predicted_image=predicted_image)
