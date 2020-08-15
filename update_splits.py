from mnist_nvae.problem import evaluate_datasets
from mnist_nvae.datastream import evaluate_datastreams


if __name__ == '__main__':
    evaluate_datasets(frozen=False)
    evaluate_datastreams(frozen=False)
