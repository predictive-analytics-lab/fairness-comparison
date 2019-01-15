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
from algorithms.FairGridSearch import FairGridSearch
from algorithms.universalgp.UGPAlgorithm import UGP, UGPDemPar, UGPEqOpp
from algorithms.universalgp.ULRAlgorithm import ulr, ulr_dem_par, ulr_eq_opp
from algorithms.gpytorch.gpyt_algorithm import GPyT, GPyTDemPar, GPyTEqOdds

from metrics.DIBinary import DIBinary
from metrics.Accuracy import Accuracy
from metrics.MCC import MCC

# algos = []
# for tnr0 in [0.6, 0.7]:  # [0.6, 0.7, 0.8, 0.9, 1.0]:
#     for tnr1 in [0.6, 0.7]:  # [0.6, 0.7, 0.8, 0.9, 1.0]:
#         for tpr in [0.6, 0.7]:  # [0.6, 0.7, 0.8, 0.9, 1.0]:
#             algos.append(UGPEqOpp(s_as_input=True, tnr0=tnr0, tnr1=tnr1, tpr0=tpr, tpr1=tpr))
#             algos.append(UGPEqOpp(s_as_input=False, tnr0=tnr0, tnr1=tnr1, tpr0=tpr, tpr1=tpr))
# ALGORITHMS = algos

ALGORITHMS = [
    UGP(s_as_input=True),
    UGP(s_as_input=False),
    UGPDemPar(s_as_input=True),
    UGPDemPar(s_as_input=True, average_prediction=True),
    UGPDemPar(s_as_input=False),
    UGPDemPar(s_as_input=True, target_mode=UGPDemPar.MIN),
    UGPDemPar(s_as_input=True, average_prediction=True, target_mode=UGPDemPar.MIN),
    UGPDemPar(s_as_input=False, target_mode=UGPDemPar.MIN),
    UGPDemPar(s_as_input=True, target_mode=UGPDemPar.MAX),
    UGPDemPar(s_as_input=True, average_prediction=True, target_mode=UGPDemPar.MAX),
    UGPDemPar(s_as_input=False, target_mode=UGPDemPar.MAX),
    UGPEqOpp(s_as_input=True),
    UGPEqOpp(s_as_input=True, average_prediction=True),
    UGPEqOpp(s_as_input=False),
    SVM(),
    GaussianNB(),
    LogisticRegression(),
    DecisionTree(),
    CaldersAlgorithm(),
    KamishimaAlgorithm(),
    FeldmanAlgorithm(SVM()),
    FeldmanAlgorithm(GaussianNB()),       # Feldman
    FeldmanAlgorithm(LogisticRegression()),
    FeldmanAlgorithm(DecisionTree()),
    ZafarAlgorithmFairness(),
    ZafarAlgorithmAccuracy(),
    ZafarEOAlgorithmOpp(),
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
