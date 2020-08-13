from pydantic import BaseModel
from typing import Callable, List, Optional
from itertools import product
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from mnist_nvae.problem import Audio, Example
from mnist_nvae import architecture, tools


class Prediction(BaseModel):
    predicted_speech: torch.Tensor

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def audio(self) -> Audio:
        return Audio(self.predicted_speech.cpu().numpy())

    def representation(self, example=None) -> Image:
        if example is None:
            return self.audio().representation()
        else:
            return Image.fromarray(np.concatenate([
                np.array(example.representation()),
                np.array(self.audio().representation()),
            ]))

    @property
    def _repr_png_(self):
        return self.representation()._repr_png_


class PredictionBatch(BaseModel):
    predicted_speech: torch.Tensor
    kl_losses: Optional[List[torch.Tensor]]

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def stack_clean_speech(self, examples):
        return torch.from_numpy(
            np.stack([
                example.clean_speech.waveform for example in examples
            ])
        ).to(self.predicted_speech)

    def loss(self, examples, hack=[1]):
        hack[0] *= 0.9999
        return (
            -self.snr(examples)
            + 0.05 * sum(self.kl_losses) * 10 ** (-4 * hack[0])
        )

    def mse(self, examples):
        return F.mse_loss(
            input=self.predicted_speech,
            target=self.stack_clean_speech(examples),
        )

    def __len__(self):
        return len(self.predicted_speech)

    def __getitem__(self, index):
        return Prediction(
            predicted_speech=self.predicted_speech[index],
        )

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

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
