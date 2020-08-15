from itertools import product
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pydantic import BaseModel

from mnist_nvae.problem import settings


class Example(BaseModel):
    image: Image.Image

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def representation(self):
        return self.image

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_

    def augment(self, augmenter):
        image = Image.fromarray(
            augmenter.augment(image=np.array(self.image))
        )
        return Example(image=image)
