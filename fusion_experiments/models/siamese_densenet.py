#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch

from models import registry


__all__ = [
    'SiameseDenseNet',
]


def _make_normalization(normalization, inputs):
    """
    Make normalization layer.

    Args:
        normalization (str): Type of normalization layer.
        inputs        (int): Input channel size.

    Returns:
        (torch.nn.Module): Normalization module.
    """
    if normalization == 'batchnorm2d':
        return torch.nn.BatchNorm2d(inputs)
    else:
        raise Exception('{} normalization does not exist.'.format(normalization))


def _make_activation(activation):
    """
    Make activation layer.

    Args:
        activation (str): Type of activation.

    Returns:
        (torch.nn.Module): Activation module.
    """
    if activation == 'relu':
        return torch.nn.ReLU(inplace=True)
    else:
        raise Exception('{} activation does not exist.'.format(activation))


def _make_conv_block(inputs,
                     outputs,
                     kernel_size,
                     padding=0,
                     normalization='batchnorm2d',
                     activation='relu'):
    """



    """
    block = torch.nn.Sequential(
        _make_normalization(normalization, inputs),
        _make_activation(activation),
        torch.nn.Conv2d(inputs, outputs, kernel_size=kernel_size, padding=padding, bias=False))
    return block


def _make_preprocess(inputs,
                     outputs,
                     normalization='batchnorm2d',
                     activation='relu'):
    """
    Make pre-processing block.

    Args:
        inputs        (int): Input channel size.
        outputs       (int): Output channel size.
        normalization (str): Type of normalization; defaults to batch norm.
        activation    (str): Type of activation; defaults to relu.

    Returns:
        (torch.nn.Module): Pre-processing block.
    """
    preprocess = torch.nn.Sequential(
        torch.nn.Conv2d(inputs, outputs, kernel_size=7, stride=2, padding=3, bias=False),
        _make_normalization(normalization, outputs),
        _make_activation(activation),
        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    return preprocess


class _DenseLayer(torch.nn.Module):

    def __init__(self,
                 inputs,
                 growth_rate=32,
                 bn_size=4,
                 drop_rate=0):
        """
        Initialization.

        Args:
            inputs      (int):   Number of input features.
            growth_rate (int):   Number of layers to add per layer.
            bn_size     (int):   Multiplicative factor for number of
                                 bottleneck layers; i.e., `bn_size * k`
                                 features in the bottleneck layer.
            drop_rate   (float): Dropout rate after each dense block.
        """
        super(_DenseLayer, self).__init__()
        self.conv1 = _make_conv_block(inputs, bn_size * growth_rate, 1)
        self.conv2 = _make_conv_block(bn_size * growth_rate, growth_rate, 3, padding=1)
        self.dropout = torch.nn.Dropout2d(p=drop_rate)

    def forward(self, xs):
        """
        Perform a forward pass of the module, then
        construct dense features for the next block.

        Args:
            xs (torch.Tensor): [B x F1 x H x W] feature tensor.

        Returns:
            (torch.Tensor): [B x (F1 + F2) x H x W] feature tensor.
        """
        features = self.dropout(self.conv2(self.conv1(xs)))
        return torch.cat([xs, features], dim=1)


class _DenseBlock(torch.nn.Sequential):

    def __init__(self,
                 inputs,
                 num_layers,
                 bn_size,
                 growth_rate,
                 drop_rate):
        """
        Initialization.

        Args:

        """
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer{}'.format(i + 1), _DenseLayer(
                inputs + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate,
            ))


class _Transition(torch.nn.Sequential):

    def __init__(self, inputs, outputs):
        """
        Initialization.

        Args:

        """
        super(_Transition, self).__init__()
        self.add_module('conv', _make_conv_block(inputs, outputs, 1))
        self.add_module('pool', torch.nn.AvgPool2d(kernel_size=2, stride=2))


class SiameseDenseNet(torch.nn.Module):
    """
    DenseNet-BC model, modified for multi-frame fusion.
    """

    def __init__(self,
                 fusion_index=-1,
                 growth_rate=32,
                 block_config=(6, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000):
        """
        Initialization.

        Args:
            fusion_index      (int):   Block index after which to do fusion.
            growth_rate       (int):   Number of layers to add per layer.
            block_config      (list):  Number of layers per pooling block.
            num_init_features (int):   Number of filters in first conv layer.
            bn_size           (int):   Multiplicative factor for number of
                                       bottleneck layers; i.e., `bn_size * k`
                                       features in the bottleneck layer.
            drop_rate         (float): Dropout rate after each dense block.
            num_classes       (list):  Number of classes.
        """
        super(SiameseDenseNet, self).__init__()

        self.frontal = torch.nn.ModuleList([])
        self.lateral = torch.nn.ModuleList([])

        # 1. Add pre-processing blocks.
        if fusion_index < 0:
            self.frontal.append(_make_preprocess(6, num_init_features))
        else:
            self.frontal.append(_make_preprocess(3, num_init_features))
            self.lateral.append(_make_preprocess(3, num_init_features))

        num_features = num_init_features
        if fusion_index ==  0:
            num_features = 2 * num_features

        # 2. Add DenseBlocks and TransitionBlocks.
        for i, num_layers in enumerate(block_config, 1):
            if fusion_index >= i:
                self.lateral.append(_DenseBlock(
                    num_features,
                    num_layers,
                    bn_size,
                    growth_rate,
                    drop_rate,
                ))

            self.frontal.append(_DenseBlock(
                num_features,
                num_layers,
                bn_size,
                growth_rate,
                drop_rate,
            ))

            num_features = num_features + num_layers * growth_rate
            if fusion_index >= i:
                self.lateral.append(_Transition(num_features, num_features // 2))
            self.frontal.append(_Transition(num_features, num_features // 2))

            num_features = num_features // 2
            if fusion_index == i:
                num_features = 2 * num_features

        # 3. Add the classifier.
        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Conv1d(num_features, num_classes, (1, 1)),
        )

    def forward(self, xs1, xs2):
        """
        Perform a forward pass of the module.

        Args:
            xs1 (torch.Tensor): [B x 3 x H x W] image tensor; frontal image.
            xs2 (torch.Tensor): [B x 3 x H x W] image tensor; lateral image.

        Returns:
            (torch.Tensor): [B x K] logits.
        """
        if not torch.is_tensor(xs2):
            assert(torch.is_tensor(xs1))
            xs2 = torch.zeros_like(xs1)

        if not torch.is_tensor(xs1):
            assert(torch.is_tensor(xs2))
            xs1 = torch.zeros_like(xs2)

        B, F, H, W = xs1.size()

        # 1. Extract before-fusion features.
        for i in range(len(self.lateral)):
            xs1 = self.frontal[i](xs1)
            xs2 = self.lateral[i](xs2)

        # 2. Fusion!
        xs = torch.cat([xs1, xs2], dim=1)

        # 3. Extract after-fusion features.
        for i in range(len(self.lateral), len(self.frontal)):
            xs = self.frontal[i](xs)

        # 4. Classify!
        out = self.classifier(xs).view(B, -1)
        return out


@registry.MODELS.register('siamese_densenet121')
def make_siamese_densenet121(config):
    """



    """
    model = SiameseDenseNet(
        fusion_index=config['model']['fusion_index'],
        num_classes=len(config['general']['classes']),
    )
    return model

