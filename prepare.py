import argparse
from pathlib import Path
import shutil
import pandas as pd
import torchvision
from tqdm import tqdm

from vae import problem


CACHE_ROOT = Path('cache')


def image_path(directory, index):
    return directory / f'{index}.png'


def save_images(dataset, directory):
    original_size = (178, 218)
    crop_size = 148
    left = (original_size[0] - crop_size) // 2
    top = (original_size[1] - crop_size) // 2

    for index, (image, _) in enumerate(tqdm(dataset)):
        (
            image.crop((left, top, left + crop_size, top + crop_size))
            .resize((problem.settings.WIDTH, problem.settings.HEIGHT))
            .save(image_path(directory, index))
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    dataset = torchvision.datasets.CelebA(CACHE_ROOT, split='all')

    shutil.rmtree('prepared')
    directory = Path('prepared')
    directory.mkdir(parents=True)
    save_images(dataset, directory)
    # shutil.rmtree(CACHE_ROOT)
