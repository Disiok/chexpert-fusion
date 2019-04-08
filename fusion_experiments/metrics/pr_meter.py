#
#
#


import numpy as np
import sklearn.metrics


__all__ = [
    'PRMeter',
]


class PRMeter(object):

    def __init__(self, classes):
        """
        Initialization.

        Args:
            classes (list): List of classes.
        """
        super(PRMeter, self).__init__()

        self.classes = classes
        self.reset()

    def reset(self):
        """
        Reset PR meter.
        """
        self.masks = np.ndarray((0, len(self.classes)), dtype=np.float32)
        self.scores = np.ndarray((0, len(self.classes)), dtype=np.float32)
        self.targets = np.ndarray((0, len(self.classes)), dtype=np.int64)

    def add_predictions(self, masks, scores, targets):
        """
        Add predictions.

        Args:
            masks   (np.array): [N x K] binary array.
            scores  (np.array): [N x K] array of prediction scores.
            targets (np.array): [N] array of target labels.
        """
        self.masks = np.concatenate([self.masks, masks], axis=0)
        self.scores = np.concatenate([self.scores, scores], axis=0)
        self.targets = np.concatenate([self.targets, targets], axis=0)

    def get_scores(self, class_id):
        """
        Retrieve (masked) prediction scores.

        Args:
            class_id (int): Index of class.

        Returns:
            (np.array): [N] array of prediction scores.
        """
        mask = (self.masks[:, class_id] == 1.)
        return self.scores[:, class_id][mask]

    def get_targets(self, class_id):
        """
        Retrieve (masked) target labels.

        Args:
            class_id (int): Index of class.

        Returns:
            (np.array): [N] array of target labels.
        """
        mask = (self.masks[:, class_id] == 1.)
        return self.targets[:, class_id][mask]

    def values(self):
        """
        Retrieve AP metrics.

        Returns:
            (dict): Map from class_name to AP.
        """
        metrics = {}
        total_ap = 0.
        num_classes = 0

        for class_id, class_name in enumerate(self.classes):
            scores = self.get_scores(class_id)
            targets = self.get_targets(class_id)

            if len(scores) == 0:
                metrics[class_name] = 0.0
                continue

            ap = sklearn.metrics.average_precision_score(targets, scores)
            if ap != ap:  # If ap is nan
                metrics[class_name] = 0.0
            else:
                total_ap += ap
                num_classes += 1
                metrics[class_name] = ap

        metrics['mean'] = total_ap / (num_classes + 1e-8)
        return metrics

