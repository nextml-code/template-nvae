import numpy as np
from datastream import Datastream

from mnist_nvae.datastream import (
    evaluate_datastreams, augmenter
)
from mnist_nvae.problem import settings


def GradientDatastream():
    dataset = evaluate_datastreams()['gradient'].dataset
    
    augmenter_ = augmenter()
    return (
        Datastream.merge([
            Datastream(dataset.subset(
                lambda df: df['class_name'] == class_name
            ))
            for class_name in settings.CLASS_NAMES
        ])
        .map(lambda example: example.augment(augmenter_))
    )
