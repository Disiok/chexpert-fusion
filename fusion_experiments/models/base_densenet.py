from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict

from models import registry, fusions, backbones

__all__ = [
    "BaseDenset",
]

class BaseDenset(nn.Module):
    def __init__(self, num_classes=10, num_init_features=64, normalization=None, activation=None, view='frontal'):
        super(BaseDenset, self).__init__()
        assert view in ['frontal', 'lateral']
        self.view = view
        self.feature_net = backbones.DenseStream(num_init_features=num_init_features)

        # Final batch norm
        self.final_norm = nn.BatchNorm2d(self.feature_net.num_features[-1])

        # Linear layer
        self.classifier = nn.Linear(self.feature_net.num_features[-1], num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, frontal, lateral):
        if self.view == 'frontal':
            features = frontal
        elif self.view == 'lateral':
            features = lateral
        else:
            raise NotImplementedError
        
        features = self.feature_net(features)
        features = self.final_norm(features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)

        return out

@registry.MODELS.register('base_densenet121')
def make_base_densenet121(config):
    """

    """
    model = BaseDenset(
        num_classes=len(config['general']['classes']),
        normalization=config['model']['normalization'],
        activation=config['model']['activation'],
        view=config['model']['view'],
    )
    return model

