from torch import optim
from torch import nn

import config
import models


def get_experiment():
    if config.model_name == 'DenseNet':
        model = models.DenseNet(num_init_features=64, 
                                growth_rate=32, 
                                block_config=(6, 12, 24, 16), 
                                num_classes=len(config.class_names))
    else:
        raise NotImplementedError

    if config.optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    else:
        raise NotImplementedError

    if config.loss_name == 'CrossEntropyLoss':
        loss_criterion = nn.CrossEntropyLoss(reduction='none')
    elif config.loss_name == 'BCELoss':
        loss_criterion = nn.BCELoss(reduction='none')
    else:
        raise NotImplementedError

    return model, optimizer, loss_criterion
