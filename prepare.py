import argparse
from pathlib import Path
import pandas as pd
import torchvision
from tqdm import tqdm

from vae import problem


CACHE_ROOT = 'cache'


def image_path(directory, index):
    return directory / f'{index}.png'


def save_images(dataset, directory):
    size = (problem.settings.WIDTH, problem.settings.HEIGHT)
    for index, (image, _) in enumerate(tqdm(dataset)):
        image.resize(size).save(image_path(directory, index))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    dataset = torchvision.datasets.CelebA(CACHE_ROOT, split='all', download=True)

    directory = Path('prepared')
    directory.mkdir(parents=True)
    save_images(dataset, directory)
