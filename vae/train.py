import os
from functools import partial
import numpy as np
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import ignite
import logging
import workflow
from workflow import json
from workflow.functional import starcompose
from workflow.torch import set_seeds, module_eval
from workflow.ignite import worker_init
from workflow.ignite.handlers.learning_rate import (
    LearningRateScheduler, warmup, cyclical
)
from datastream import Datastream
from simple_pid import PID
from pydantic import BaseModel
from typing import List

from vae import datastream, architecture, metrics

torch.backends.cudnn.benchmark = True


class KLWeightController(BaseModel):
    weights: np.ndarray
    targets: np.ndarray
    pids: List[PID]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, weights, targets):
        super().__init__(
            weights=np.array(weights, dtype=np.float32),
            targets=np.array(targets, dtype=np.float32),
            pids=KLWeightController.new_pids(weights, targets),
        )

    @staticmethod
    def new_pids(weights, targets):
        pids = [
            PID(
                # P = -1 gives oscillations, this is on purpose
                -1, -0.1, -0.0,
                setpoint=np.log10(target),
                auto_mode=False,
            )
            for target in targets
        ]
        for pid, weight in zip(pids, weights):
            pid.set_auto_mode(True, last_output=np.log10(weight))
        return pids

    def update_(self, kl_losses):
        if len(self.pids) != len(kl_losses):
            raise ValueError('Expected same number of kl as pid controllers')

        for index, (pid, kl) in enumerate(zip(self.pids, kl_losses)):
            self.weights[index] = 10 ** pid(np.log10(kl.item()), dt=1)
        return self.weights

    def state_dict(self):
        return dict(weights=self.weights)

    def load_state_dict(self, state_dict):
        self.weights = state_dict['weights']
        self.pids = KLWeightController.new_pids(self.weights, self.targets)

    def zero_(self):
        self.weights = np.full_like(self.weights, 1e-3)
        self.pids = KLWeightController.new_pids(self.weights, self.targets)
    
    def map_(self, fn):
        self.weights = fn(self.weights)
        self.pids = KLWeightController.new_pids(self.weights, self.targets)


def train(config):
    set_seeds(config['seed'])

    device = torch.device('cuda' if config['use_cuda'] else 'cpu')

    model = architecture.Model(config).to(device)
    optimizer = torch.optim.Adamax(
        model.parameters(), lr=config['learning_rate']
    )
    kl_weight_controller = KLWeightController(
        weights=sum([
            [1e3 for _ in range(level_size)]
            for level_index, level_size in enumerate(model.level_sizes)
        ], list()),
        # weights=sum([
        #     [
        #         1e2 * 2 ** level_index
        #         for _ in range(level_size)
        #     ]
        #     for level_index, level_size in enumerate(model.level_sizes)
        # ], list()),
        targets=sum([
            [
                10 ** (-2.5 - level_index * 0.2)
                # 10 ** (-3)
                for _ in range(level_size)
            ]
            for level_index, level_size in enumerate(model.level_sizes)
        ], list()),
    )

    train_state = dict(
        model=model,
        optimizer=optimizer,
        kl_weight_controller=kl_weight_controller,
    )

    if os.path.exists('model'):
        print('Loading model checkpoint')
        workflow.ignite.handlers.ModelCheckpoint.load(
            train_state, 'model/checkpoints', device
        )
        workflow.torch.set_learning_rate(optimizer, config['learning_rate'])

    # kl_weight_controller.zero_()
    # kl_weight_controller.map_(
    #     lambda weights: weights * 1e-3
    # )

    n_parameters = sum([
        p.shape.numel() for p in model.parameters() if p.requires_grad
    ])
    print(f'n_parameters: {n_parameters:,}')

    def process_batch(examples):
        predictions = model.prediction(
            architecture.FeaturesBatch.from_examples(examples)
        )
        return (
            predictions,
            predictions.loss(examples, kl_weight_controller.weights),
        )

    @workflow.ignite.decorators.train(model, optimizer)
    def train_batch(engine, examples):
        # with torch.cuda.amp.autocast():
        predictions, loss = process_batch(examples)
        loss.backward()

        # if engine.state.iteration % 20 == 0 and engine.state.epoch > 5:
        if engine.state.iteration % 20 == 0:
            kl_weight_controller.update_(predictions.kl_losses)

        if engine.state.iteration % 2000 == 0:
            kl_weight_controller.map_(
                lambda weights: weights * 1e-2
            )

        return dict(
            examples=examples,
            predictions=predictions.cpu().detach(),
            loss=loss,
            kl_weights=kl_weight_controller.weights,
        )

    @workflow.ignite.decorators.evaluate(model)
    def evaluate_batch(engine, examples):
        # with torch.cuda.amp.autocast():
        predictions, loss = process_batch(examples)
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
        if 'compare' in name
    }

    trainer, evaluators, tensorboard_logger = workflow.ignite.trainer(
        train_batch,
        evaluate_batch,
        evaluate_data_loaders,
        metrics=dict(
            progress=metrics.progress_metrics(),
            train=metrics.train_metrics(),
            **{
                name: metrics.evaluate_metrics()
                for name in evaluate_data_loaders.keys()
            },
        ),
        optimizers=optimizer,
    )

    workflow.ignite.handlers.ModelScore(
        lambda: trainer.state.epoch,
        train_state,
        {
            name: metrics.evaluate_metrics()
            for name in evaluate_data_loaders.keys()
        },
        tensorboard_logger,
        config,
    ).attach(trainer, evaluators)

    def log_examples(description):
        def log_examples_(engine, logger, event_name):
            n_examples = 5
            indices = np.random.choice(
                len(engine.state.output['predictions']),
                n_examples,
                replace=False,
            )

            logger.writer.add_images(
                f'{description}/predictions',
                np.stack([
                    np.concatenate([
                        np.array(
                            engine.state.output['examples'][index]
                            .representation()
                        ),
                        np.array(
                            engine.state.output['predictions'][index]
                            .representation()
                        ),
                    ], axis=0) / 255
                    for index in indices
                ]),
                trainer.state.epoch,
                dataformats='NHWC',
            )

            with torch.no_grad(), module_eval(model) as eval_model:
                std_samples = [
                    eval_model.generated(16, prior_std)
                    for prior_std in np.linspace(0.5, 1.5, num=11)
                ]

            logger.writer.add_images(
                f'{description}/samples',
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

            with torch.no_grad(), module_eval(model) as eval_model:
                partial_samples = [
                    eval_model.partially_generated(
                        architecture.FeaturesBatch.from_examples(
                            [
                                engine.state.output['examples'][index]
                                for index in indices
                            ]
                        ).image_batch,
                        sample=[
                            index == sample_index
                            for index in range(config['levels'])
                        ],
                        prior_std=0.7,
                    )
                    for sample_index in range(config['levels'])
                ]

            logger.writer.add_images(
                f'{description}/partially_sampled',
                np.concatenate([
                    np.stack([
                        np.array(sample.representation())
                        for sample in samples
                    ])
                    for samples in partial_samples
                ], axis=1) / 255,
                trainer.state.epoch,
                dataformats='NHWC',
            )                    

        return log_examples_
    
    tensorboard_logger.attach(
        trainer,
        log_examples('train'),
        ignite.engine.Events.EPOCH_COMPLETED,
    )

    for name, evaluator in evaluators.items():
        tensorboard_logger.attach(
            evaluator, log_examples(name), ignite.engine.Events.EPOCH_COMPLETED
        )

    trainer.run(
        data=(
            datastream.GradientDatastream()
            .data_loader(
                batch_size=config['batch_size'],
                num_workers=config['n_workers'],
                n_batches_per_epoch=config['n_batches_per_epoch'],
                worker_init_fn=partial(worker_init, config['seed'], trainer),
                collate_fn=tuple,
            )
        ),
        max_epochs=config['max_epochs'],
    )
