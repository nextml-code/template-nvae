from typing import List
import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel

from mnist_nvae import problem


class FeaturesBatch(BaseModel):
    image_batch: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    @staticmethod
    def from_examples(examples: List[problem.Example]):
        return FeaturesBatch.from_images(
            [example.image for example in examples]
        )

    @staticmethod
    def from_images(images: List[Image.Image]):
        return FeaturesBatch(
            image_batch=torch.as_tensor(
                [np.array(image) for image in images]
            ).float()
        )
