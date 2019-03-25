import numpy as np
import torch
from cross_sectional_models import CrossSectionalDensetNet


def main():
    model = CrossSectionalDensetNet(num_classes=10)

    frontal_images = torch.FloatTensor(np.zeros((1, 3, 320, 320)))
    lateral_images = torch.FloatTensor(np.ones((1, 3, 320, 320)))

    out = model(frontal_images, lateral_images)

    import ipdb; ipdb.set_trace()
    

if __name__ == '__main__':
    main()
