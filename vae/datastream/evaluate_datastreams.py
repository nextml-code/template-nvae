from datastream import Datastream

from vae import datastream


def evaluate_datastreams():
    datasets = datastream.datasets()
    return {
        name: Datastream(dataset).take(256)
        for name, dataset in datastream.datasets().items()
    }
