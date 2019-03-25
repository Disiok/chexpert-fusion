#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch


__all__ = [
    'EmptyImageTransform',
]


class EmptyImageTransform(object):

    def __init__(self, xsize, ysize):
        """


        """
        super(EmptyImageTransform, self).__init__()
        self.xsize = xsize
        self.ysize = ysize

    def __call__(self, image):
        """



        """
        if torch.is_tensor(image):
            return image

        return torch.zeros((3, self.xsize, self.ysize), dtype=torch.float)

