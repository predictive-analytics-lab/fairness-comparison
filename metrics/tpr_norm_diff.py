import math

from metrics.Metric import Metric


class TPRNormDiff(Metric):
    """
    This metric calculates the normalized TPR difference.

    Multiple protected classes are treated as one large group, so that this compares the privileged
    class to all non-privileged classes as a group.
    """
    def __init__(self):
        Metric.__init__(self)
        self.name = 'TPRNormDiff'
        self.is_tnr = False

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        unprotected_tpr, protected_tpr = calc_tpr_protected(
            actual, predicted, sensitive, unprotected_vals, positive_pred, is_tnr=self.is_tnr)
        tpr_norm_diff = 1.0
        if unprotected_tpr > 0:
            tpr_norm_diff = (unprotected_tpr - protected_tpr) / (unprotected_tpr + protected_tpr)
        if unprotected_tpr == 0.0 and protected_tpr == 0.0:
            tpr_norm_diff = 0.0
        return tpr_norm_diff

    def is_better_than(self, val1, val2):
        dist1 = math.fabs(val1)
        dist2 = math.fabs(val2)
        return dist1 <= dist2


class TNRNormDiff(TPRNormDiff):
    """This metric calculates the TNR ratio."""
    def __init__(self):
        super().__init__()
        self.name = 'TNRNormDiff'
        self.is_tnr = True


def calc_tpr_protected(actual, predicted, sensitive, unprotected_vals, positive_pred, is_tnr=False):
    """
    Returns P(C=YES|Y=YES, sensitive=privileged) and P(C=YES|Y=YES, sensitive=not privileged)
    in that order where C is the predicited classification and where all not privileged values are
    considered equivalent. Assumes that predicted and sensitive have the same lengths.

    If `is_tnr` is true, this actually computes the TNRs instead of the TPRs.
    """
    unprotected_true_pos = 0.0
    unprotected_pos_label = 0.0
    protected_true_pos = 0.0
    protected_pos_label = 0.0
    for protected_val, predicted_val, label in zip(sensitive, predicted, actual):
        if not is_tnr:  # do TPR in this case
            criterion = str(label) == str(positive_pred)  # prediction should have been positive
        else:
            criterion = str(label) != str(positive_pred)  # prediction should have been negative
        if criterion:
            if str(predicted_val) == str(label):  # prediction was correct
                if protected_val in unprotected_vals:
                    unprotected_true_pos += 1
                else:
                    protected_true_pos += 1

            if protected_val in unprotected_vals:
                unprotected_pos_label += 1
            else:
                protected_pos_label += 1

    unprotected_tpr = 0.0
    if unprotected_pos_label > 0:
        unprotected_tpr = unprotected_true_pos / unprotected_pos_label
    protected_tpr = 0.0
    if protected_pos_label > 0:
        protected_tpr = protected_true_pos / protected_pos_label

    return unprotected_tpr, protected_tpr
