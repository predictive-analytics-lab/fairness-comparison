from algorithms.zafar.ZafarAlgorithm import ZafarAlgorithmBaseline, ZafarAlgorithmAccuracy, ZafarAlgorithmFairness
from algorithms.kamishima.KamishimaAlgorithm import KamishimaAlgorithm
from algorithms.kamishima.CaldersAlgorithm import CaldersAlgorithm
from algorithms.feldman.FeldmanAlgorithm import FeldmanAlgorithm
from algorithms.baseline.SVM import SVM
from algorithms.baseline.DecisionTree import DecisionTree
from algorithms.baseline.GaussianNB import GaussianNB
from algorithms.baseline.LogisticRegression import LogisticRegression
from algorithms.ParamGridSearch import ParamGridSearch
from algorithms.universalgp.UniversalGPAlgorithm import UniversalGPAlgorithm

from metrics.DIAvgAll import DIAvgAll
from metrics.Accuracy import Accuracy
from metrics.MCC import MCC

ALGORITHMS = [CaldersAlgorithm(), UniversalGPAlgorithm(s_as_input=True)]

# ALGORITHMS = [UniversalGPAlgorithm(s_as_input=True)]     # baseline
#              SVM(),
#              GaussianNB(),
#              LogisticRegression(),
#              DecisionTree(),

#   KamishimaAlgorithm(),                                          # Kamishima
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

