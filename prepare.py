import argparse
from pathlib import Path
import pandas as pd
import torchvision

from mnist_nvae import problem

# more realistic to have class names without a natural 1-to-1 mapping with
# integers
CLASS_TO_NAME = dict(zip(range(10), [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
]))
CACHE_ROOT = 'cache'


def image_path(directory, index):
    return directory / f'{index}.png'


def save_images(dataset, directory):
    for index, (image, _) in enumerate(dataset):
        image.save(image_path(directory, index))


def save_labels(dataset, image_directory, csv_path):
    (
        pd.DataFrame(dict(
            index=range(len(dataset)),
            number=dataset.targets,
        ))
        .assign(
            class_name=lambda df: (
                df['number'].map(CLASS_TO_NAME)
            ),
            image_path=lambda df: df['index'].apply(
                lambda index: image_path(image_directory, index)
            )
        )
        [['index', 'image_path', 'class_name']]
        .to_csv(csv_path)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    dataset_and_directory = [
        (
            problem.settings.TRAIN_CSV,
            Path('prepared/train'),
            torchvision.datasets.MNIST(
                CACHE_ROOT, train=True, download=True
            ),
        ),
        (
            problem.settings.TEST_CSV,
            Path('prepared/test'),
            torchvision.datasets.MNIST(
                CACHE_ROOT, train=False, download=True
            ),
        ),
    ]

    # saving images and labels to disk to simulate a more realistic use case
    # can preprocess (e.g. resize) images before training this way
    for csv_path, directory, dataset in dataset_and_directory:
        directory.mkdir(parents=...)
        save_images(dataset, directory)
        save_labels(dataset, directory, csv_path)
