from datastream import Datastream

from vae import problem


def datasets():
    # not splitting train into gradient and early_stopping since the latter
    # is not used
    datasets = problem.datasets()
    return dict(
        gradient=datasets['train'],
        compare=datasets['compare'],
    )
