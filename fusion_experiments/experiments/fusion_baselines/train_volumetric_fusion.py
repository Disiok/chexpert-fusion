#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import argparse

from experiments import registry
from experiments.fusion_baselines.trainer import Trainer

from config.chexpert import CHEXPERT_CLASSES
from config.chexpert import PAPER_TRAINING_CLASSES


__all__ = [
    'train_volumetric_fusion',
]


CLASSES = {
    'default': CHEXPERT_CLASSES,
    'paper'  : PAPER_TRAINING_CLASSES,
}


@registry.EXPERIMENTS.register('train_volumetric_fusion')
def train_volumetric_fusion(argv):
    """
    Run fusion baseline experiments.

    Args:
        argv (list): List of command line strings.
    """
    parser = argparse.ArgumentParser(description='Fusion Baselines')

    # General
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--evaluate-once', action='store_true', default=False)
    parser.add_argument('--use-test-set', action='store_true', default=False)
    parser.add_argument('--cuda-benchmark', action='store_true', default=False)

    # Dataset
    parser.add_argument('--dataset-class', type=str, default='PairedOnlyCustomSplit')
    parser.add_argument('--label-class', type=str, default='default')
    parser.add_argument('--map-unobserved-to-negative', action='store_true', default=False)
    parser.add_argument('--train-data', type=str, required=True)
    parser.add_argument('--train-batch-size', type=int, default=1)
    parser.add_argument('--val-data', type=str, required=True)
    parser.add_argument('--val-batch-size', type=int, default=1)
    parser.add_argument('--shuffle', action='store_true', default=False)

    # Training
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--fusion-index', type=int, default=-1)
    parser.add_argument('--is-big', action='store_true', default=False)
    parser.add_argument('--use-2d-conv', action='store_true', default=False)
    parser.add_argument('--normalization', type=str, default='batchnorm2d')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--criterion', type=str, default='bce_loss')

    # IO
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--val-frequency', type=int, default=4800)
    parser.add_argument('--log-frequency', type=int, default=1000)
    parser.add_argument('--save-predictions', action='store_true', default=False)

    args = parser.parse_args(argv)
    print(args)

    general_configuration = {
        'seed'          : args.seed,
        'num_gpus'      : args.num_gpus,
        'use_cuda'      : args.num_gpus > 0,
        'num_epochs'    : args.num_epochs,
        'checkpoint'    : args.checkpoint,
        'cuda_benchmark': args.cuda_benchmark,
        'classes'       : CLASSES[args.label_class],
    }

    model_configuration = {
        'class'        : 'volumetric_fusion_densenet121',
        'fusion_index' : args.fusion_index,
        'normalization': args.normalization,
        'activation'   : args.activation,
        'use_2d_conv'  : args.use_2d_conv,
        'is_big'       : args.use_2d_conv,
    }

    criterion_configuration = {
        'class': args.criterion,
        'gamma': 1,
    }

    optimizer_configuration = {
        'class'        : 'adam',
        'milestones'   : [],
        'learning_rate': args.learning_rate,
        'lr_decay'     : 0.1,
    }

    train_data_configuration = {
        'class'         : args.dataset_class,
        'map_unobserved_to_negative': args.map_unobserved_to_negative,
        'batch_size'    : args.train_batch_size,
        'num_workers'   : args.num_workers,
        'pin_memory'    : args.num_gpus > 0,
        'shuffle'       : args.shuffle,
        'dataset_path'  : args.train_data,
        'image_size'    : 320,
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std' : [0.299, 0.224, 0.224],
    }

    val_data_configuration = copy.deepcopy(train_data_configuration)
    val_data_configuration['batch_size'] = args.val_batch_size
    val_data_configuration['dataset_path'] = args.val_data

    io_configuration = {
        'outdir'          : args.outdir,
        'log_frequency'   : args.log_frequency,
        'val_frequency'   : args.val_frequency,
        'save_predictions': args.save_predictions,
    }

    configuration = {
        'general'   : general_configuration,
        'model'     : model_configuration,
        'criterion' : criterion_configuration,
        'optimizer' : optimizer_configuration,
        'train_data': train_data_configuration,
        'val_data'  : val_data_configuration,
        'io'        : io_configuration,
    }

    trainer = Trainer(configuration)
    trainer.evaluate(args.use_test_set) if args.evaluate_once else trainer.train()

