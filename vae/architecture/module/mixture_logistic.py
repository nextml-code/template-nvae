import torch
import torch.distributions as D
from torch.distributions.transformed_distribution import (
    TransformedDistribution
)
from torch.distributions.transforms import (
    SigmoidTransform, AffineTransform
)


class Logistic(TransformedDistribution):
    def __init__(self, loc, scale):
        super().__init__(
            D.Uniform(torch.zeros_like(loc), 1),
            [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)]
        )
        self.loc = loc

    @property
    def mean(self):
        return self.loc


def MixtureLogistic(logits, loc, scale):
    return D.MixtureSameFamily(
        D.Categorical(logits=logits),
        Logistic(loc, scale),
    )
