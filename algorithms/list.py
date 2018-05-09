from algorithms.zafar.ZafarAlgorithm import ZafarAlgorithmBaseline, ZafarAlgorithmAccuracy, ZafarAlgorithmFairness
from algorithms.kamishima.KamishimaAlgorithm import KamishimaAlgorithm
from algorithms.kamishima.CaldersAlgorithm import CaldersAlgorithm
from algorithms.feldman.FeldmanAlgorithm import FeldmanAlgorithm
from algorithms.baseline.SVM import SVM
from algorithms.baseline.DecisionTree import DecisionTree
from algorithms.baseline.GaussianNB import GaussianNB
from algorithms.baseline.LogisticRegression import LogisticRegression
from algorithms.ParamGridSearch import ParamGridSearch
from algorithms.universalgp.FairGPAlgorithm import GPAlgorithm, FairGPAlgorithm

from metrics.DIAvgAll import DIAvgAll
from metrics.Accuracy import Accuracy
from metrics.MCC import MCC

ALGORITHMS = [
    GPAlgorithm(s_as_input=False),
    GPAlgorithm(s_as_input=True),
    FairGPAlgorithm(s_as_input=True, target_acceptance=.1),
    FairGPAlgorithm(s_as_input=True, target_acceptance=.2),
    FairGPAlgorithm(s_as_input=True, target_acceptance=.3),
    FairGPAlgorithm(s_as_input=True, target_acceptance=.4),
    FairGPAlgorithm(s_as_input=True, target_acceptance=.5),
    FairGPAlgorithm(s_as_input=True, target_acceptance=.6),
    FairGPAlgorithm(s_as_input=True, target_acceptance=.7),
    FairGPAlgorithm(s_as_input=True, target_acceptance=.8),
    FairGPAlgorithm(s_as_input=True, target_acceptance=.9),
    FairGPAlgorithm(s_as_input=False, target_acceptance=.1),
    FairGPAlgorithm(s_as_input=False, target_acceptance=.2),
    FairGPAlgorithm(s_as_input=False, target_acceptance=.3),
    FairGPAlgorithm(s_as_input=False, target_acceptance=.4),
    FairGPAlgorithm(s_as_input=False, target_acceptance=.5),
    FairGPAlgorithm(s_as_input=False, target_acceptance=.6),
    FairGPAlgorithm(s_as_input=False, target_acceptance=.7),
    FairGPAlgorithm(s_as_input=False, target_acceptance=.8),
    FairGPAlgorithm(s_as_input=False, target_acceptance=.9),
    # SVM(),
    # GaussianNB(),
    # LogisticRegression(),
    # DecisionTree(),
    # CaldersAlgorithm(),
    # KamishimaAlgorithm(),
    # FeldmanAlgorithm(SVM()), FeldmanAlgorithm(GaussianNB()),       # Feldman
    # FeldmanAlgorithm(LogisticRegression()), FeldmanAlgorithm(DecisionTree()),
    # ZafarAlgorithmFairness(),
    # ZafarAlgorithmAccuracy(),
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

