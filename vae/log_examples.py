import numpy as np
import torch
from workflow.torch import module_eval

from vae import architecture


def log_examples(description, trainer, model):
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
                for prior_std in np.linspace(0.4, 1.1, num=8)
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
                        for index in range(model.levels)
                    ],
                    prior_std=0.7,
                )
                for sample_index in range(model.levels)
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
