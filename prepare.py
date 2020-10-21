import argparse
from joblib import Parallel, delayed
from pathlib import Path
import shutil
import torchvision
from tqdm import tqdm

from vae import problem


CACHE_ROOT = Path('cache')


def image_path(directory, index):
    return directory / f'{index}.png'


def prepare_and_save(image, directory, index, left, top, crop_size):
    return (
        image.crop((left, top, left + crop_size, top + crop_size))
        .resize((problem.settings.WIDTH, problem.settings.HEIGHT))
        .save(image_path(directory, index))
    )


def save_images(dataset, directory, n_jobs):
    original_size = (178, 218)
    crop_size = 148
    left = (original_size[0] - crop_size) // 2
    top = (original_size[1] - crop_size) // 2

    Parallel(n_jobs)(
        delayed(prepare_and_save)(
            image, directory, index, left, top, crop_size
        )
        for index, (image, _) in enumerate(tqdm(dataset))
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', default=4, type=int)
    args = parser.parse_args()

    dataset = torchvision.datasets.CelebA(
        CACHE_ROOT, split='all', download=True
    )

    directory = Path('prepared')
    directory.mkdir(parents=True)
    save_images(dataset, directory, args.n_jobs)
    shutil.rmtree(CACHE_ROOT)
