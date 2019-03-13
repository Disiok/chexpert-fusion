import os
import numpy as np
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from PIL import Image

import config


class CheXpertDataset(Dataset):
    """
    CheXpert dataset of lung x-ray scans
    """
    def __init__(self, mode='train', transforms=None):
        """
        NOTE(suo): Only train and validation splits
        """
        assert mode in ['train', 'val'], 'Invalid mode'

        self.root_dir = config.root_dir
        self.transforms = transforms

        self.dataset_df = pd.read_csv(os.path.join(self.root_dir, '{}.csv'.format(mode)))
        replacement_func = (lambda x: config.uncertainty_mode if x == -1 else x)
        self.dataset_df = self.dataset_df.applymap(replacement_func)

        self.image_names = self.dataset_df[config.path_name].as_matrix()
        self.masks = self.dataset_df[config.class_names].notnull().as_matrix()
        self.labels = self.dataset_df.fillna(0)[config.class_names].as_matrix()

    def __len__(self):
        """
        Length of the dataset
        """
        return len(self.dataset_df)
    
    def __getitem__(self, idx):
        """
        Get data, label, and mask (for unobserved labels)
        """
        data = self._load_image(self.image_names[idx])
        label = self.labels[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32)

        if self.transforms:
            data = self.transforms(data)
        return (data, label, mask)

    def _load_image(self, image_name):
        # NOTE(suo): Take the parent dir here due to weird convention used in the dataframe
        parent_dir = os.path.dirname(self.root_dir)
        image_path = os.path.join(parent_dir, image_name)
        image = Image.open(image_path)
        image = image.convert('RGB')
        return image
    

if __name__ == '__main__':
    dataset = CheXpertDataset()
    print(dataset[0])