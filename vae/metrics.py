import numpy as np
import torch
import ignite


def progress_metrics():
    return dict(
        batch_loss=ignite.metrics.RunningAverage(
            output_transform=lambda output: output['loss'],
            epoch_bound=False,
            alpha=1e-7,
        ),
    )


def train_metrics():
    return dict(
        loss=ignite.metrics.RunningAverage(
            output_transform=lambda output: output['loss'],
            epoch_bound=False,
            alpha=1e-7,
        ),
        log_prob=ignite.metrics.RunningAverage(
            output_transform=lambda output: (
                output['predictions'].log_prob(output['examples'])
            ),
            epoch_bound=False,
            alpha=1e-7,
        ),
        kl=ignite.metrics.RunningAverage(
            output_transform=lambda output: (
                torch.tensor(output['predictions'].kl_losses)
            ),
            epoch_bound=False,
            # alpha=1e-7,
            alpha=0.7,
        ),
        kl_weights=ignite.metrics.RunningAverage(
            output_transform=lambda output: (
                torch.tensor(output['kl_weights'])
            ),
            epoch_bound=False,
            alpha=1e-7,
        ),
    )


def evaluate_metrics():
    return dict(
        loss=ignite.metrics.Average(
            lambda output: output['loss']
        ),
        log_prob=ignite.metrics.Average(lambda output: (
            output['predictions'].log_prob(output['examples'])
        )),
        kl=ignite.metrics.Average(lambda output: (
            torch.tensor(output['predictions'].kl_losses)
        )),
    )
