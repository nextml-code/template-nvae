from datastream import Datastream

from vae import datastream
from vae.datastream import augmenter


def GradientDatastream():
    augmenter_ = augmenter()
    return (
        Datastream(datastream.datasets()['gradient'])
        .map(lambda example: example.augment(augmenter_))
    )
