from pathlib import Path
from PIL import Image
import pandas as pd
from datastream import Dataset

from vae import problem


def dataframe():
    return (
        pd.DataFrame(dict(
            image_path=problem.settings.data_path.glob('*.png'),
        ))
        .assign(
            key=lambda df: df['image_path'].apply(
                lambda image_path: image_path.stem
            )
        )
    )


def dataset(dataframe):
    return (
        Dataset.from_dataframe(dataframe)
        .map(lambda row: row['image_path'])
        .map(Path)
        .map(Image.open)
        .map(lambda image: problem.Example(image=image))
        .cache('key')
    )


def datasets(frozen=True):
    return (
        dataset(dataframe())
        .split(
            key_column='key',
            proportions=dict(train=0.8, compare=0.2),
            seed=177,
        )
    )
