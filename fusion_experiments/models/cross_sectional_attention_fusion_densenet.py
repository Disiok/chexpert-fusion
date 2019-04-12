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


class CrossSectionalAttentionFusionDenseNet(nn.Module):
    def __init__(self,
                 num_classes=10,
                 num_init_features=64,
                 block_config=(6, 12, 24, 16),
                 fusion_index=0,
                 fusion_operator=fusions.CrossSectionalAttentionFusion):
        super(CrossSectionalAttentionFusionDenseNet, self).__init__()
        assert fusion_index in range(len(block_config) + 1)
        self.fusion_index = fusion_index

        self.frontal_stream = backbones.DenseStream(num_init_features=num_init_features, block_config=block_config)
        self.lateral_stream = backbones.DenseStream(num_init_features=num_init_features, block_config=block_config[:fusion_index])

        WIDTH = int(320 // 4)
        self.input_fusion = fusion_operator(num_init_features, WIDTH)
        self.block_fusions = nn.ModuleList([])

        for i, num_features in enumerate(self.lateral_stream.num_features):
            if i != len(self.lateral_stream.num_features) - 1:
                WIDTH = int(WIDTH // 2)
            self.block_fusions.append(fusion_operator(num_features, WIDTH))

        # Final batch norm and linear layer
        self.final_norm_frontal = nn.BatchNorm2d(self.frontal_stream.num_features[-1])
        self.classifier_frontal = nn.Linear(self.frontal_stream.num_features[-1], num_classes)

        if fusion_index == 4:
            self.final_norm_lateral = nn.BatchNorm2d(self.lateral_stream.num_features[-1])
            self.classifier_lateral = nn.Linear(self.lateral_stream.num_features[-1], num_classes)

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
        frontal_features = self.final_norm_frontal(frontal_features)
        frontal_out = F.relu(frontal_features, inplace=True)
        frontal_out = F.adaptive_avg_pool2d(frontal_out, (1, 1)).view(frontal_features.size(0), -1)
        frontal_out = self.classifier_frontal(frontal_out)
        return frontal_out


@registry.MODELS.register('cross_sectional_attention_fusion_densenet121')
def make_cross_sectional_attention_fusion_densenet121(config):
    """


    """
    FUSION_OPERATORS = {
        'cross_sectional_attention'            : fusions.CrossSectionalAttentionFusion,
        'cross_sectional_attention_correlation': fusions.CrossSectionalAttentionFusionCorrelation,
        'cross_sectional_attention_mlp'        : fusions.CrossSectionalAttentionFusionMLP,
    }

    model = CrossSectionalAttentionFusionDenseNet(
        num_classes=len(config['general']['classes']),
        fusion_index=config['model']['fusion_index'],
        fusion_operator=FUSION_OPERATORS[config['model']['fusion_operator']],
    )
    return model

