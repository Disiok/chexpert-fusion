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
    "VolumetricFusionDenseNet",
]


class VolumetricFusionDenseNet(nn.Module):
    def __init__(self, num_classes=10, num_init_features=64, normalization=None, activation=None):
        super(VolumetricFusionDenseNet, self).__init__()

        self.front_stream = backbones.DenseStream(num_init_features=num_init_features)
        self.lateral_stream = backbones.DenseStream(block_config=())

        self.input_fusion = fusions.VolumetricFusion(64)

        # Final batch norm
        self.final_norm = nn.BatchNorm2d(self.front_stream.num_features[-1])

        # Linear layer
        self.classifier = nn.Linear(self.front_stream.num_features[-1], num_classes)

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
        frontal_features = self.front_stream.features(frontal)
        lateral_features = self.lateral_stream.features(lateral)

        # Input fusion
        frontal_features, _ = self.input_fusion(frontal_features, lateral_features)

        # Only run front stream after fusion
        for frontal_block in self.front_stream.blocks:  
            frontal_features = frontal_block(frontal_features)

        # Classification
        features = self.final_norm(frontal_features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)

        return out

@registry.MODELS.register('volumetric_fusion_densenet121')
def make_volumetric_densenet121(config):
    """

    """
    model = VolumetricFusionDenseNet(
        num_classes=len(config['general']['classes']),
        normalization=config['model']['normalization'],
        activation=config['model']['activation'],
    )
    return model

