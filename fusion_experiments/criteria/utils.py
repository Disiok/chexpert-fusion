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
        masked_losses = losses * mask
        masked_mean = torch.sum(masked_losses, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
        loss = torch.mean(masked_mean)
        
        return loss


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

