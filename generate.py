import argparse
import os
import logging
import torch
from workflow import json
import os
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
import ignite
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger
import workflow
from workflow.torch import set_seeds, module_train, module_eval
from workflow.ignite import worker_init
from workflow.ignite.handlers.learning_rate import (
    LearningRateScheduler, warmup, cyclical
)
from workflow.ignite.handlers.progress_bar import ProgressBar

from vae import datastream, architecture, metrics

torch.backends.cudnn.benchmark = True


from vae import train

logging.getLogger('ignite').setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=40, type=int)
    parser.add_argument('--n_batches_per_epoch', default=50, type=int)
    parser.add_argument('--n_epochs', default=40, type=int)
    parser.add_argument('--prior_std', default=0.85, type=float)

    config = vars(parser.parse_args())
    config.update(
        seed=1,
        use_cuda=torch.cuda.is_available(),
        run_id=os.getenv('RUN_ID'),
    )

    set_seeds(config['seed'])

    device = torch.device('cuda' if config['use_cuda'] else 'cpu')

    model = architecture.Model(config).to(device)
    train_state = dict(model=model)

    print('Loading model checkpoint')
    workflow.ignite.handlers.ModelCheckpoint.load(
        train_state, 'model/checkpoints', device
    )

    @torch.no_grad()
    @module_train(model)
    def update_batch_norms(engine, _):
        model.generated(config['batch_size'], config['prior_std'])

    @torch.no_grad()
    @module_eval(model)
    def log_examples(engine, logger, event_name):
        std_samples = [
            model.generated(16, prior_std)
            for prior_std in np.linspace(0.1, 1.1, num=11)
        ]

        logger.writer.add_images(
            'samples',
            np.stack([np.concatenate([
                np.concatenate([
                    np.array(sample.representation())
                    for sample in samples
                ], axis=1)
                for samples in std_samples
            ], axis=0)]) / 255,
            trainer.state.epoch,
            dataformats='NHWC',
        )

    trainer = ignite.engine.Engine(update_batch_norms)

    tensorboard_logger = TensorboardLogger(log_dir='tb')
    tensorboard_logger.attach(
        trainer,
        log_examples,
        ignite.engine.Events.EPOCH_COMPLETED,
    )

    ProgressBar(desc='update_batch_norms').attach(trainer)

    trainer.run(
        range(config['n_batches_per_epoch']),
        max_epochs=config['n_epochs']
    )
