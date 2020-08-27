from vae.problem import evaluate_datasets
from vae.datastream import evaluate_datastreams


if __name__ == '__main__':
    evaluate_datasets(frozen=False)
    evaluate_datastreams(frozen=False)
