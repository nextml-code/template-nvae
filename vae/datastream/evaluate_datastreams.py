import numpy as np
import pandas as pd
from datastream import Datastream

from vae import problem, splits


def evaluate_datastreams():
    evaluate_datasets = problem.evaluate_datasets()
    return {
        split_name: Datastream(dataset)
        for split_name, dataset in dict(
            gradient=evaluate_datasets['train'],
            compare=evaluate_datasets['compare'],
        ).items()
    }
