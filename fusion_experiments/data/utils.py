#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import torch
import torchvision
import torch.utils.data

from torch.utils.data.dataloader import default_collate
from data.datasets.paired_chexpert import PairedCheXpertDataset, PairedOnlyCheXpertDataset, PairedOnlyCustomSplit 
from data.transforms.empty_image_transform import EmptyImageTransform
from data.transforms.pil_image_transform import PILImageTransform

from config.chexpert import CHEXPERT_CLASSES, PAPER_TRAINING_CLASSES


__all__ = [
    'make_dataloader',
]


def _same_dimension(a):
    """


    """
    return lambda b: torch.is_tensor(b) and b.size() == a.size()


def _custom_collate(batch):
    """



    """
    batch = filter(lambda x: x is not None, batch)
    if len(batch) <= 0:
        return None

    collated = {}
    for key in batch[0].keys():
        batch_items = [x[key] for x in batch]
        if all(map(_same_dimension(batch_items[0]), batch_items)):
            collated[key] = default_collate(batch_items)
        elif isinstance(batch_items[0], list):
            collated[key] = batch_items
        elif isinstance(batch[0], dict):
            collated[key] = batch_items
        else:
            collated[key] = default_collate(batch_items)

    return collated


def make_dataloader(config, mode):
    """



    """
    assert(mode in ['train', 'valid']), '{} is not valid'.format(mode)
    dataset_key = 'train_data' if mode == 'train' else 'val_data'

    transforms = torchvision.transforms.Compose([
        PILImageTransform(config[dataset_key]['image_size'],
                          config[dataset_key]['image_size'],
                          config[dataset_key]['normalize_mean'],
                          config[dataset_key]['normalize_std']),
        EmptyImageTransform(config[dataset_key]['image_size'],
                            config[dataset_key]['image_size']),
    ])

    # TODO(suo): Migrate this to use registry
    dataset_class = {
        'PairedCheXpertDataset': PairedCheXpertDataset,
        'PairedOnlyCheXpertDataset': PairedOnlyCheXpertDataset,
        'PairedOnlyCustomSplit': PairedOnlyCustomSplit
    }[config[dataset_key]['class']]


    # TODO(suo): Migrate this to use registry
    label_classes = {
        'default': CHEXPERT_CLASSES,
        'paper': PAPER_TRAINING_CLASSES,
    }[config['general']['classes']]


    dataset = dataset_class(
        config[dataset_key]['dataset_path'],
        mode,
        label_classes,
        transforms
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config[dataset_key]['batch_size'],
        shuffle=config[dataset_key]['shuffle'],
        num_workers=config[dataset_key]['num_workers'],
        pin_memory=config[dataset_key]['pin_memory'],
        collate_fn=_custom_collate,
    )
    return dataloader

