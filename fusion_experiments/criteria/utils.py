#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch

from criteria import registry
from criteria.focal_loss import BinaryFocalLoss


__all__ = [
    'make_bce_loss',
    'make_focal_loss',
]


class BCEWithLogitsLoss(torch.nn.Module):

    def forward(self, logits, targets, mask):
        """
        Compute binary cross-entropy loss with mask.

        Args:
            logits  (torch.Tensor): [N x K] logits.
            targets (torch.Tensor): [N x K] binary labels.
            mask    (torch.Tensor): [N x K] binary mask.

        Returns:
            (torch.Tensor): Scalar loss.
        """
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        losses = torch.masked_select(losses, mask.eq(1.))
        return torch.mean(losses)


@registry.CRITERIA.register('bce_loss')
def make_bce_loss(config):
    """


    """
    return BCEWithLogitsLoss()


@registry.CRITERIA.register('focal_loss')
def make_focal_loss(config):
    """


    """
    return BinaryFocalLoss(config['criterion']['gamma'])

