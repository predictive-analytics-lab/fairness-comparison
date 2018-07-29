from algorithms.zafar.ZafarAlgorithm import ZafarAlgorithmBaseline, ZafarAlgorithmAccuracy, ZafarAlgorithmFairness
from algorithms.zafar.ZafarEOAlgorithm import ZafarEOAlgorithmBaseline, ZafarEOAlgorithmOpp, ZafarEOAlgorithmOdd
from algorithms.kamishima.KamishimaAlgorithm import KamishimaAlgorithm
from algorithms.kamishima.CaldersAlgorithm import CaldersAlgorithm
from algorithms.feldman.FeldmanAlgorithm import FeldmanAlgorithm
from algorithms.baseline.SVM import SVM
from algorithms.baseline.DecisionTree import DecisionTree
from algorithms.baseline.GaussianNB import GaussianNB
from algorithms.baseline.LogisticRegression import LogisticRegression
from algorithms.ParamGridSearch import ParamGridSearch
from algorithms.universalgp.UGPAlgorithm import UGP, UGPDemPar, UGPEqOpp

from metrics.DIAvgAll import DIAvgAll
from metrics.Accuracy import Accuracy
from metrics.MCC import MCC

ALGORITHMS = [
    # UGP(s_as_input=True),
    # UGP(s_as_input=False),
    # UGPDemPar(s_as_input=True),
    # UGPDemPar(s_as_input=True, average_prediction=True),
    # UGPDemPar(s_as_input=False),
    # UGPDemPar(s_as_input=True, target_acceptance=0.3),
    # UGPDemPar(s_as_input=False, target_acceptance=0.3),
    # UGPDemPar(s_as_input=True, target_acceptance=0.4),
    # UGPDemPar(s_as_input=False, target_acceptance=0.4),
    # UGPDemPar(s_as_input=True, target_acceptance=0.5),
    # UGPDemPar(s_as_input=False, target_acceptance=0.5),
    # UGPDemPar(s_as_input=True, target_mode=UGPDemPar.MIN),
    # UGPDemPar(s_as_input=True, average_prediction=True, target_mode=UGPDemPar.MIN),
    # UGPDemPar(s_as_input=False, target_mode=UGPDemPar.MIN),
    # UGPDemPar(s_as_input=True, target_mode=UGPDemPar.MAX),
    # UGPDemPar(s_as_input=True, average_prediction=True, target_mode=UGPDemPar.MAX),
    # UGPDemPar(s_as_input=False, target_mode=UGPDemPar.MAX),
    UGPDemPar(s_as_input=True, precision_target=1.0),
    UGPDemPar(s_as_input=False, precision_target=1.0),
    # UGPDemPar(s_as_input=True, precision_target=0.9),
    # UGPDemPar(s_as_input=False, precision_target=0.9),
    UGPDemPar(s_as_input=True, precision_target=0.8),
    UGPDemPar(s_as_input=False, precision_target=0.8),
    # UGPDemPar(s_as_input=True, precision_target=0.7),
    # UGPDemPar(s_as_input=False, precision_target=0.7),
    UGPDemPar(s_as_input=True, precision_target=0.6),
    UGPDemPar(s_as_input=False, precision_target=0.6),
    # UGPDemPar(s_as_input=True, precision_target=0.5),
    # UGPDemPar(s_as_input=False, precision_target=0.5),
    UGPDemPar(s_as_input=True, precision_target=0.4),
    UGPDemPar(s_as_input=False, precision_target=0.4),
    # UGPDemPar(s_as_input=True, precision_target=0.3),
    # UGPDemPar(s_as_input=False, precision_target=0.3),
    UGPDemPar(s_as_input=True, precision_target=0.2),
    UGPDemPar(s_as_input=False, precision_target=0.2),
    # UGPDemPar(s_as_input=True, precision_target=0.1),
    # UGPDemPar(s_as_input=False, precision_target=0.1),
    # UGPEqOpp(s_as_input=True),
    # UGPEqOpp(s_as_input=True, average_prediction=True),
    # UGPEqOpp(s_as_input=False),
    # SVM(),
    # GaussianNB(),
    # LogisticRegression(),
    # DecisionTree(),
    # CaldersAlgorithm(),
    # KamishimaAlgorithm(),
    # FeldmanAlgorithm(SVM()),
    # FeldmanAlgorithm(GaussianNB()),       # Feldman
    # FeldmanAlgorithm(LogisticRegression()),
    # FeldmanAlgorithm(DecisionTree()),
    # ZafarAlgorithmFairness(),
    # ZafarAlgorithmAccuracy(),
    # ZafarEOAlgorithmOpp(),
    # ZafarEOAlgorithmOdd(),
]

# ALGORITHMS = [UniversalGPAlgorithm(s_as_input=True)]     # baseline


#    KamishimaAlgorithm(),                                        # Kamishima
#    CaldersAlgorithm(),                                            # Calders
#    ZafarAlgorithmBaseline(),                                      # Zafar
#    ZafarAlgorithmFairness(),
# #   ZafarAlgorithmAccuracy(),
#    ParamGridSearch(KamishimaAlgorithm(), Accuracy()),             # Kamishima params
#    ParamGridSearch(KamishimaAlgorithm(), DIAvgAll()),
#    FeldmanAlgorithm(SVM()), FeldmanAlgorithm(GaussianNB()),       # Feldman
#    FeldmanAlgorithm(LogisticRegression()), FeldmanAlgorithm(DecisionTree()),
#    ParamGridSearch(FeldmanAlgorithm(SVM()), DIAvgAll()),          # Feldman params
#    ParamGridSearch(FeldmanAlgorithm(SVM()), Accuracy()),
#    ParamGridSearch(FeldmanAlgorithm(GaussianNB()), DIAvgAll()),
#    ParamGridSearch(FeldmanAlgorithm(GaussianNB()), Accuracy())
