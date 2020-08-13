from itertools import product
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from typing import Tuple, Optional, List

from mnist_nvae import problem


def text_(draw, text, x, y, fill='black', outline='white', size=12):
    font = ImageFont.load_default()

    for x_shift, y_shift in product([-1, 0, 1], [-1, 0, 1]):
        draw.text((x + x_shift, y + y_shift), text, font=font, fill=outline)

    draw.text((x, y), text, font=font, fill=fill)


def class_index(class_name):
    return problem.settings.CLASS_NAMES.index(class_name)


class Prediction(BaseModel):
    predicted_image: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def image(self):
        return Image.fromarray(np.uint8(
            self.predicted_image.squeeze(0).cpu().numpy()
        ))

    def representation(self):
        return self.image().copy().resize((256, 256))

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
        return torch.from_numpy(
            np.stack([
                np.array(example.image) for example in examples
            ])
        ).to(self.predicted_image)

    def loss(self, examples):
        return (
            self.mse(examples)
            + sum(self.kl_losses)
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
