import torch
import torch.distributions as D
from torch.distributions.transformed_distribution import (
    TransformedDistribution
)
from torch.distributions.transforms import (
    SigmoidTransform, AffineTransform
)


def Logistic(loc, scale):
    return TransformedDistribution(
        # D.Uniform(0, 1).expand(loc.shape),
        D.Uniform(torch.zeros_like(loc), 1),
        [SigmoidTransform().inv, AffineTransform(loc=loc, scale=scale)]
    )


def MixtureLogistic(logits, loc, scale):
    return D.MixtureSameFamily(
        D.Categorical(logits=logits),
        Logistic(loc, scale),
    )
