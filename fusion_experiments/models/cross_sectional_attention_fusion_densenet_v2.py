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
    "CrossSectionalAttentionFusionDenseNet",
]

class CrossSectionalAttentionFusionDenseNetV2(nn.Module):
    def __init__(self, num_classes=10, num_init_features=64, block_config=(6, 12, 24, 16), normalization=None, activation=None, fusion_index=0):
        super(CrossSectionalAttentionFusionDenseNetV2, self).__init__()
        assert fusion_index in range(len(block_config) + 1)
        self.fusion_index = fusion_index

        self.frontal_stream = backbones.DenseStream(num_init_features=num_init_features, block_config=block_config)
        self.lateral_stream = backbones.DenseStream(num_init_features=num_init_features, block_config=block_config[:fusion_index])

        self.input_fusion = fusions.CrossSectionalAttentionFusionV2(num_init_features)
        self.block_fusions = nn.ModuleList([fusions.CrossSectionalAttentionFusionV2(num_features) for num_features in self.lateral_stream.num_features])

        # Final batch norm
        self.final_norm = nn.BatchNorm2d(self.frontal_stream.num_features[-1])

        # Linear layer
        self.classifier = nn.Linear(self.frontal_stream.num_features[-1], num_classes)

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
        frontal_features = self.frontal_stream.features(frontal)
        lateral_features = self.lateral_stream.features(lateral)

        # Input fusion
        frontal_features, lateral_features = self.input_fusion(frontal_features, lateral_features)

        # Block fusions
        for frontal_block, lateral_block, block_fusion in zip(self.frontal_stream.blocks[:self.fusion_index], self.lateral_stream.blocks, self.block_fusions):
            frontal_features, lateral_features = frontal_block(frontal_features), lateral_block(lateral_features)
            frontal_features, lateral_features = block_fusion(frontal_features, lateral_features)

        # Only run front stream after fusion
        for frontal_block in self.frontal_stream.blocks[self.fusion_index:]:
            frontal_features = frontal_block(frontal_features)

        # Classification
        features = self.final_norm(frontal_features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)

        return out

@registry.MODELS.register('cross_sectional_attention_fusion_densenet121_v2')
def make_cross_sectional_attention_fusion_densenet121(config):
    """

    """
    model = CrossSectionalAttentionFusionDenseNetV2(
        num_classes=len(config['general']['classes']),
        normalization=config['model']['normalization'],
        activation=config['model']['activation'],
        fusion_index=config['model']['fusion_index'],
    )
    return model

