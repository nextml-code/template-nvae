import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from functools import partial
import numpy as np
import random
import argparse
import torch
import torch.nn.functional as F
import ignite
from ignite.engine import Engine, Events
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger, OutputHandler
)
import logging
import workflow
from workflow import json
from workflow.functional import starcompose
from workflow.torch import set_seeds
from workflow.ignite import worker_init, evaluator
from workflow.ignite.handlers.learning_rate import (
    LearningRateScheduler, warmup, cyclical
)
from workflow.ignite.handlers import (
    EpochLogger,
    MetricsLogger,
    ProgressBar
)
from datastream import Datastream

from mnist_nvae import datastream, architecture, metrics

logging.getLogger('ignite').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def evaluate(config):
    device = torch.device('cuda' if config['use_cuda'] else 'cpu')

    model = architecture.Model().to(device)

    train_state = dict(model=model)

    print('Loading model checkpoint')
    workflow.ignite.handlers.ModelCheckpoint.load(
        train_state, 'model/checkpoints', device
    )


    @workflow.ignite.decorators.evaluate(model)
    def evaluate_batch(engine, examples):
        predictions = model.predicted(tuple(example.image for example in examples))
        loss = predictions.loss(tuple(example.class_name for example in examples))
        return dict(
            examples=examples,
            predictions=predictions.cpu().detach(),
            loss=loss,
        )


    evaluate_data_loaders = {
        f'evaluate_{name}': datastream.data_loader(
            batch_size=config['eval_batch_size'],
            num_workers=config['n_workers'],
            collate_fn=tuple,
        )
        for name, datastream in datastream.evaluate_datastreams().items()
    }

    tensorboard_logger = TensorboardLogger(log_dir='tb')

    for desciption, data_loader in evaluate_data_loaders.items():
        engine = evaluator(
            evaluate_batch, desciption, metrics.evaluate_metrics(), tensorboard_logger
        )
        engine.run(data=data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--n_workers', default=2, type=int)

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

    evaluate(config)

