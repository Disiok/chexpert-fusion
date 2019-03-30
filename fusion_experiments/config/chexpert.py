#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


__all__ = [
    'CHEXPERT_CLASSES',
]

CHEXPERT_CLASSES = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices',
]

# NOTE(suo): No Finding is mutually exclusive will all other classes
#            except Support Devices

TRAINING_CLASSES = [
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
]

PAPER_TRAINING_CLASSES = [
    'Cardiomegaly',
    'Edema',
    'Consolidation',
    'Atelectasis',
    'Pleural Effusion',
]