#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import numpy as np


__all__ = [
    'to_device',
    'to_numpy',
    'to_torch',
]


def to_device(xs, device):
    """


    """
    if torch.is_tensor(xs):
        return xs.to(device)
    elif isinstance(xs, list):
        return map(lambda x: to_device(x, device), xs)
    elif isinstance(xs, dict):
        return {k: to_device(v) for (k, v) in xs.items()}
    else:
        return xs


def to_numpy(xs):
    """


    """
    if torch.is_tensor(xs):
        return xs.cpu().numpy()
    elif isinstance(xs, list):
        return map(lambda x: to_numpy(x), xs)
    elif isinstance(xs, dict):
        return {k: to_numpy(v) for (k, v) in xs.items()}
    else:
        return xs


def to_torch(xs):
    """


    """
    if isinstance(xs, np.ndarray):
        return torch.from_numpy(xs)
    elif isinstance(xs, list):
        return map(lambda x: to_torch(x), xs)
    elif isinstance(xs, dict):
        return {k: to_torch(v) for (k, v) in xs.items()}
    else:
        return xs
