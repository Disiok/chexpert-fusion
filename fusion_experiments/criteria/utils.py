#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch

from criteria import registry


__all__ = [
    'make_cross_entropy',
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
        return torch.mean(losses * mask)


@registry.CRITERIA.register('bce_loss')
def make_bce_loss(config):
    """


    """
    return BCEWithLogitsLoss()

