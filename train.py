import argparse
import os
import logging
from functools import partial
import torch
from workflow import json
from workflow.functional import starcompose
from workflow.ignite import worker_init
from workflow.ignite.handlers.learning_rate import (
    LearningRateScheduler, warmup, cyclical
)

from vae import train

logging.getLogger('ignite').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--n_batches_per_epoch', default=200, type=int)
    parser.add_argument('--n_batches_per_step', default=1, type=int)
    parser.add_argument('--patience', type=float, default=40)
    parser.add_argument('--n_workers', default=0, type=int)
    parser.add_argument('--levels', default=5, type=int)

    try:
        __IPYTHON__
        args = parser.parse_known_args()[0]
    except NameError:
        args = parser.parse_args()

    config = vars(args)
    config.update(
        seed=1,
        use_cuda=torch.cuda.is_available(),
        run_id=os.getenv('RUN_ID'),
    )

    json.write(config, 'config.json')

    train(config)
