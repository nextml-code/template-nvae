from pathlib import Path
from PIL import Image
import pandas as pd
from datastream import Dataset

from mnist_nvae import problem


def MnistDataset(dataframe):
    return (
        Dataset.from_dataframe(dataframe)
        .map(lambda row: (
            Path(row['image_path']),
            row['class_name'],
        ))
        .starmap(lambda image_path, class_name: problem.Example(
            image=Image.open(image_path),
            class_name=class_name,
        ))
    )


def evaluate_datasets():
    # TODO: consider showing the standard case where there is no
    # predetermined split
    train_df = pd.read_csv(problem.settings.TRAIN_CSV)
    test_df = pd.read_csv(problem.settings.TEST_CSV)

    return dict(
        train=MnistDataset(train_df),
        compare=MnistDataset(test_df),
    )