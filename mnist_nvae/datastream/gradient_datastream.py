import numpy as np
from datastream import Datastream

from mnist_nvae.datastream import (
    evaluate_datastreams, augmenter
)


def GradientDatastream():    
    augmenter_ = augmenter()
    return (
        evaluate_datastreams()['gradient']
        .map(lambda example: example.augment(augmenter_))
    )
