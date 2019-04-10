#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import torch.utils.data
import numpy as np
import pandas as pd

from PIL import Image
from lib.utils import to_torch


__all__ = [
    'PairedMIMICDataset',
    'PairedOnlyMIMICDataset',
]


def load_studies(root, mode, class_names, paired_only, map_unobserved_to_negative):
    """
    Load a list of studies.

    Args:
        root        (str):  Path to root directory.
        mode        (str):  One of `train` or `val`.
        class_names (list): List of class names.
        paired_only (bool): Whether to filter to only studies with paired images

    Returns:
        (list): List of items containing:
            patient  (str):      Patient identifier.
            study_id (str):      Study ID for the patient.
            frontal  (str):      Path to frontal image, or None.
            lateral  (str):      Path to lateral image, or None.
            masks    (np.array): [14] binary mask array.
            labels   (np.array): [14] binary label array.
    """
    dataset_df = pd.read_csv(os.path.join(root, '{}.csv'.format(mode)))
    if map_unobserved_to_negative:
        dataset_df = dataset_df.fillna(0.)
    dataset_df = dataset_df.applymap(lambda x: None if x == -1. else x)

    masks = dataset_df[class_names].notnull().as_matrix().astype(np.float32)
    labels = dataset_df.fillna(0)[class_names].as_matrix().astype(np.float32)
    paths = dataset_df['path'].values
    views = dataset_df['view'].values

    patient_to_studies = {}
    for ind in range(len(paths)):
        _, patient, study_id, image_fn = paths[ind].split('/')
        if (patient, study_id) not in patient_to_studies:
            patient_to_studies[(patient, study_id)] = {}
            patient_to_studies[(patient, study_id)]['patient'] = patient
            patient_to_studies[(patient, study_id)]['study_id'] = study_id
            patient_to_studies[(patient, study_id)]['mask'] = masks[ind]
            patient_to_studies[(patient, study_id)]['labels'] = labels[ind]
            patient_to_studies[(patient, study_id)]['frontal'] = None
            patient_to_studies[(patient, study_id)]['lateral'] = None

        is_frontal = views[ind] == 'frontal'
        image_key = 'frontal' if is_frontal else 'lateral'
        patient_to_studies[(patient, study_id)][image_key] = paths[ind]
    
    studies = patient_to_studies.values()
    
    if paired_only:
        studies = filter(lambda study: study['frontal'] and study['lateral'], studies)

    return studies


class PairedMIMICDataset(torch.utils.data.Dataset):
    """
    CheXpert dataset with paired images.
    """

    def __init__(self,
                 root,
                 mode,
                 classes,
                 transforms,
                 map_unobserved_to_negative,
                 paired_only=False):
        """
        Initialization.

        Args:
            root       (str): Path to root directory.
            mode       (str): One of `train` or `val`.
            transforms (list):
        """
        super(PairedMIMICDataset, self).__init__()

        # NOTE(suo): We currently only have validation set for MIMIC-CXR
        mode='valid'

        self.root = root
        self.transforms = transforms
        self.studies = load_studies(root, mode, classes, paired_only, map_unobserved_to_negative)

    def _load_image(self, image_fn):
        """
        Load image into memory.

        Args:
            image_fn (str): Path to image.

        Returns:
            (PIL.Image, optional): The image, or None if `image_fn == None`.
        """
        if not image_fn:
            return None

        image_fn = os.path.join(self.root, image_fn)
        image = Image.open(image_fn).convert('RGB')
        return image

    def __getitem__(self, index):
        """
        Retrieve data for index.

        Args:
            index (int):

        Returns:
            (dict): Dictionary containing:
                patient  (str):          Patient identifier.
                study_id (int):          Study ID for the patient.
                frontal  (PIL.Image):    Frontal image, or None.
                lateral  (PIL.Image):    Lateral image, or None.
                masks    (torch.Tensor): [14] binary mask array.
                labels   (torch.Tensor): [14] binary label array.
        """
        study = self.studies[index]

        patient = study['patient']
        study_id = study['study_id']
        frontal_image = self._load_image(study['frontal'])
        lateral_image = self._load_image(study['lateral'])
        masks = to_torch(study['mask'])
        labels = to_torch(study['labels'])

        if self.transforms:
            frontal_image = self.transforms(frontal_image)
            lateral_image = self.transforms(lateral_image)

        result = {
            'patient' : patient,
            'study_id': study_id,
            'frontal' : frontal_image,
            'lateral' : lateral_image,
            'mask'    : masks,
            'labels'  : labels,
        }
        return result

    def __len__(self):
        """
        Length of the dataset.
        """
        return len(self.studies)


class PairedOnlyMIMICDataset(PairedMIMICDataset):
    """
    CheXpert dataset of only studies with paired images

    NOTE(suo): The validation set is only 31 studies
    """
    def __init__(self,
                 root,
                 mode,
                 classes,
                 transforms,
                 map_unobserved_to_negative):
        super(PairedOnlyMIMICDataset, self).__init__(root, 
                                                        mode, 
                                                        classes, 
                                                        transforms, 
                                                        map_unobserved_to_negative,
                                                        paired_only=True)

class PairedOnlyCustomSplitMIMIC(PairedOnlyMIMICDataset):
    """
    MIMIC-CXR dataset of only studies with paired images
    """
    def __init__(self,
                 root,
                 mode,
                 classes,
                 transforms,
                 map_unobserved_to_negative,
                 custom_split=[30000, 30707]):
        super(PairedOnlyCustomSplitMIMIC, self).__init__(root, 
                                                    'train', 
                                                    classes, 
                                                    transforms,
                                                    map_unobserved_to_negative)
        assert custom_split[0] > 0 and custom_split[-1] < len(self.studies)
        
        if mode == 'train':
            self.studies = self.studies[:custom_split[0]]
        elif mode == 'valid':
            self.studies = self.studies[custom_split[0]:custom_split[1]]
        elif mode == 'test':
            self.studies = self.studies[custom_split[1]:]
        else:
            raise NotImplementedError
        

def main():
    class_names = [
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
        'Support Devices'
    ]

    dataset = PairedMIMICDataset(
        '/home/suo/data/CheXpert-v1.0',
        'train',
        class_names,
        None,
        True
    )

    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
