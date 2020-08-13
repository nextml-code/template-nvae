from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from workflow.torch import ModuleCompose, module_device

from mnist_nvae import architecture


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = ModuleCompose(
            lambda x: torch.stack([x], dim=1),
            architecture.Encoder(32, levels=config['levels']),
        )

        self.latent_channels = 20
        self.decoder = architecture.DecoderNVAE(
            example_features=self.encoder(torch.zeros(1, 16, 16)),
            latent_channels=self.latent_channels,
        )

        def add_sn(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if len(m._forward_pre_hooks) == 0:
                    return torch.nn.utils.spectral_norm(m)
            else:
                return m

        self.apply(add_sn)

    def forward(self, image_batch):
        image_batch = image_batch.to(module_device(self))
        features = self.encoder(image_batch)
        predicted_image, kl_losses = self.decoder(features)
        # import pdb; pdb.set_trace()
        return architecture.PredictionBatch(
            predicted_image=predicted_image,
            kl_losses=kl_losses,
        )

    def prediction(self, features_batch: architecture.FeaturesBatch):
        return self(features_batch.image_batch)

    def generated(self, n_samples):
        predicted_image = self.decoder.generated((
            n_samples,
            self.latent_channels,
            self.decoder.latent_height,
            self.decoder.latent_width,
        ))
        return architecture.PredictionBatch(
            predicted_image=predicted_image,
        )
