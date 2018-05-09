"""Positive rate"""
import numpy as np
from metrics.Metric import Metric


class PR(Metric):
    """
    Returns the positive rate for the predictions. Assumes binary classification.
    """
    def __init__(self):
        super().__init__()
        self.name = 'PR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name, unprotected_vals, positive_pred):
        return np.sum(np.array(predicted) == positive_pred) / len(predicted)
