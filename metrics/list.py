from metrics.Accuracy import Accuracy
# from metrics.BCR import BCR
from metrics.CalibrationNeg import CalibrationNeg
from metrics.CalibrationPos import CalibrationPos
from metrics.CV import CV
from metrics.DIAvgAll import DIAvgAll
from metrics.DIBinary import DIBinary
from metrics.FNR import FNR
from metrics.FPR import FPR
# from metrics.MCC import MCC
from metrics.SensitiveMetric import SensitiveMetric
from metrics.TNR import TNR
from metrics.TPR import TPR
from metrics.PR import PR
from metrics.tpr_norm_diff import TPRNormDiff, TNRNormDiff

METRICS = [Accuracy(), TPR(), TNR(),        # accuracy metrics
           DIBinary(), DIAvgAll(), TPRNormDiff(), TNRNormDiff(), CV(),           # fairness metrics
           SensitiveMetric(Accuracy), SensitiveMetric(TPR), SensitiveMetric(TNR),
           SensitiveMetric(FPR), SensitiveMetric(FNR), SensitiveMetric(PR),
           SensitiveMetric(CalibrationPos), SensitiveMetric(CalibrationNeg)]


def get_metrics(dataset, sensitive_dict, tag):
    """
    Takes a dataset object and a dictionary mapping sensitive attributes to a list of the sensitive
    values seen in the data.  Returns an expanded list of metrics based on the base METRICS.
    """
    metrics = []
    for metric in METRICS:
        metrics += metric.expand_per_dataset(dataset, sensitive_dict, tag)
    return metrics
