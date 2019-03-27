#
#
#


import numpy as np
import sklearn.metrics


__all__ = [
    'AUCMeter',
]


class AUCMeter(object):

    def __init__(self, classes):
        """


        """
        super(AUCMeter, self).__init__()

        self.classes = classes
        self.reset()

    def reset(self):
        """


        """
        self.scores = np.ndarray((0, len(self.classes)), dtype=np.float32)
        self.targets = np.ndarray((0, len(self.classes)), dtype=np.int64)

    def add_predictions(self, scores, targets):
        """

        Args:
            scores  (np.ndarray): [N x K] probability scores.
            targets (np.ndarray): [N x K] binary targets.
        """

        self.scores = np.concatenate([self.scores, scores], axis=0)
        self.targets = np.concatenate([self.targets, targets], axis=0)

    def values(self):
        """


        """
        mask = np.sum(self.targets, axis=0) != 0
        targets = self.targets[:, mask]
        scores = self.scores[:, mask]
        auc_scores = np.zeros((len(self.classes),), dtype=np.float32)

        try:
            auc_scores[mask] = sklearn.metrics.roc_auc_score(
                targets, scores, average=None)
        except ValueError as e:
            print('Encountered exception: {}.'.format(str(e)))

        metrics = {'mean': np.mean(auc_scores)}
        for (i, class_name) in enumerate(self.classes):
            metrics[class_name] = auc_scores[i]
        return metrics

