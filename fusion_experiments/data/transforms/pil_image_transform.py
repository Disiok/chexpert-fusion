#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torchvision


__all__ = [
    'PILImageTransform',
]


class PILImageTransform(object):

    def __init__(self, xsize, ysize, mean, std):
        """


        """
        super(PILImageTransform, self).__init__()
        self.xsize = xsize
        self.ysize = ysize
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((xsize, ysize)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        """


        """
        if image is not None:
            image = self.transforms(image)
        return image

