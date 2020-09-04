from itertools import product
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple, Optional, List

from vae import problem
from vae.architecture import module


class Prediction(BaseModel):
    predicted_image: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def image(self):
        return Image.fromarray(np.uint8(
            self.predicted_image
            .clamp(-1, 1)
            .permute(1, 2, 0)
            .add(1)
            .mul(255 / 2)
            .cpu()
            .numpy()
        ))

    def representation(self):
        return self.image()

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_


class PredictionBatch(BaseModel):
    logits: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor
    kl_losses: Optional[List[torch.Tensor]]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    @property
    def predicted_image(self):
        return torch.distributions.TransformedDistribution(
            module.MixtureLogistic(self.logits, self.loc, self.scale),
            [
                torch.distributions.transforms.TanhTransform(cache_size=1),
                torch.distributions.transforms.AffineTransform(0, 1.1),
            ],
        )

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, index):
        distribution = torch.distributions.TransformedDistribution(
            module.MixtureLogistic(
                self.logits[index], self.loc[index], self.scale[index]
            ),
            [
                torch.distributions.transforms.TanhTransform(),
                torch.distributions.transforms.AffineTransform(0, 1.1),
            ],
        )
        samples = distribution.sample((50,)).clamp(-1, 1)
        indices = distribution.log_prob(samples).argmax(dim=0)
        most_likely_sample = (
            samples.flatten(start_dim=1)
            [indices.flatten(), torch.arange(indices.numel())]
            .view(*samples.shape[1:])
        )
        return Prediction(predicted_image=most_likely_sample)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def stack_images(self, examples):
        return (
            torch.from_numpy(
                np.stack([
                    np.array(example.image) for example in examples
                ])
            )
            .permute(0, 3, 1, 2)
            .to(self.logits)
            / 255 * 2 - 1
        )

    def loss(self, examples, kl_weights):
        return (
            -self.log_prob(examples)
            + sum([w * kl for w, kl in zip(kl_weights, self.kl_losses)])
        )

    def log_prob(self, examples):
        return self.predicted_image.log_prob(
            self.stack_images(examples)
        ).mean()

    def cpu(self):
        return PredictionBatch(**{
            name: (
                value.cpu() if isinstance(value, torch.Tensor)
                else [v.cpu() for v in value] if type(value) == list
                else value
            )
            for name, value in super().__iter__()
        })

    def detach(self):
        return PredictionBatch(**{
            name: (
                value.detach() if isinstance(value, torch.Tensor)
                else [v.detach() for v in value] if type(value) == list
                else value
            )
            for name, value in super().__iter__()
        })
