from itertools import product
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple, Optional, List

from vae import problem


class Prediction(BaseModel):
    predicted_image: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def image(self):
        return Image.fromarray(np.uint8(
            self.predicted_image
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
    predicted_image: torch.Tensor
    kl_losses: Optional[List[torch.Tensor]]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def __len__(self):
        return len(self.predicted_image)

    def __getitem__(self, index):
        return Prediction(
            predicted_image=self.predicted_image[index],
        )

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
            .to(self.predicted_image)
            / 255 * 2 - 1
        )

    def loss(self, examples, kl_weights):
        return (
            self.mse(examples)
            + sum([w * kl for w, kl in zip(kl_weights, self.kl_losses)])
        )

    def mse(self, examples):
        return F.mse_loss(
            input=self.predicted_image,
            target=self.stack_images(examples),
        )

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
