from itertools import product
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pydantic import BaseModel

from mnist_nvae.problem import settings


def text_(draw, text, x, y, fill='black', outline='white', size=16):
    font = ImageFont.load_default()

    for x_shift, y_shift in product([-1, 0, 1], [-1, 0, 1]):
        draw.text((x + x_shift, y + y_shift), text, font=font, fill=outline)

    draw.text((x, y), text, font=font, fill=fill)


class Example(BaseModel):
    image: Image.Image
    class_name: str

    @staticmethod
    def from_mnist(image, class_name):
        image = image.copy().resize((settings.WIDTH, settings.HEIGHT))
        draw = ImageDraw.Draw(image)
        text_(draw, class_name, 4, 4)
        return Example(image=image, class_name=class_name)

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
        return Example(
            image=image,
            class_name=self.class_name
        )
