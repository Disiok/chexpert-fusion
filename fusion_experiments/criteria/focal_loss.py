#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch


__all__ = [
    'BinaryFocalLoss',
]


class BinaryFocalLoss(torch.nn.Module):

    def __init__(self, gamma, size_average=True):
        """
        Initialization.

        Args:
            gamma        (float):          Focussing paramter.
            size_average (bool, optional): Average or sum loss.
        """
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logits, targets, mask):
        """
        Compute focal binary cross-entropy loss.

        Args:
            logits  (torch.Tensor): [B x *] logits tensor.
            targets (torch.Tensor): [B x *] target tensor.
            mask    (torch.Tensor): [B x *] mask tensor.

        Returns:
            (torch.Tensor): Scalar loss.
        """
        logits = logits.view(-1)
        targets = targets.view(-1)
        mask = mask.view(-1)

        prob = torch.sigmoid(logits)
        prob_pos = prob[targets.eq(1) & mask.eq(1)]
        prob_neg = prob[targets.eq(0) & mask.eq(1)]
        loss_pos = -prob_pos.log() * torch.pow(1 - prob_pos, self.gamma)
        loss_neg = -(1 - prob_neg).log() * torch.pow(prob_neg, self.gamma)

        loss = loss_pos.sum() + loss_neg.sum()
        return loss / len(logits) if self.size_average else loss

