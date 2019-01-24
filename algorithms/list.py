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
from algorithms.gpytorch.gpyt_algorithm import GPyT, GPyTDemPar, GPyTEqOdds, GPyTCal

from metrics.DIBinary import DIBinary
from metrics.tpr_norm_diff import TPRNormDiff, TNRNormDiff
from metrics.Accuracy import Accuracy
from metrics.MCC import MCC

algos1 = []
# for tnr0 in [0.6, 0.7]:  # [0.6, 0.7, 0.8, 0.9, 1.0]:
#     for tnr1 in [0.6, 0.7]:  # [0.6, 0.7, 0.8, 0.9, 1.0]:
# for tpr in [0.6, 0.7, 0.8, 0.9, 1.0]:
#     tnr_race_False = 0.724
#     tnr_race_True = 0.702
#     tnr_sex_True = 0.724
#     tnr_sex_False = 0.744
#     algos1 += [GPyTEqOdds(s_as_input=True, tnr0=tnr_sex_True, tnr1=tnr_sex_True, tpr0=tpr, tpr1=tpr)]
#     algos1 += [GPyTEqOdds(s_as_input=False, tnr0=tnr_sex_False, tnr1=tnr_sex_False, tpr0=tpr, tpr1=tpr)]

# for target_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
#     algos1 += [GPyTDemPar(s_as_input=True, target_acceptance=target_rate)]
#     algos1 += [GPyTDemPar(s_as_input=False, target_acceptance=target_rate)]

# for tnr0 in [0.6, 0.7]:  # [0.6, 0.7, 0.8, 0.9, 1.0]:
#     for tnr1 in [0.6, 0.7]:  # [0.6, 0.7, 0.8, 0.9, 1.0]:
#         for tpr in [0.6, 0.7, 0.8, 0.9, 1.0]:
#             algos1.append(ULREqOpp(s_as_input=True, tnr0=tnr0, tnr1=tnr1, tpr0=tpr, tpr1=tpr,
#                                    l2_factor=0.1, use_bias=False))
#             algos1.append(ULREqOpp(s_as_input=False, tnr0=tnr0, tnr1=tnr1, tpr0=tpr, tpr1=tpr,
#                                    l2_factor=0.1, use_bias=False))

# for target_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
#     algos1.append(FairGridSearch(ulr_dem_par(s_as_input=True, use_bias=True, target_acceptance=target_rate), Accuracy(), DIBinary(), 5))
#     algos1.append(FairGridSearch(ulr_dem_par(s_as_input=False, use_bias=True, target_acceptance=target_rate), Accuracy(), DIBinary(), 5))

# for tpr_rate in [0.6, 0.7, 0.8, 0.9, 1.0]:
#     algos1.append(FairGridSearch(ulr_eq_opp(s_as_input=True, use_bias=True, tpr0=tpr_rate,
#                                             tpr1=tpr_rate, tnr1=0.717291, tnr0=0.717291),
#                                  Accuracy(), DIBinary(), 5))
#     algos1.append(FairGridSearch(ulr_eq_opp(s_as_input=False, use_bias=True, tpr0=tpr_rate,
#                                             tpr1=tpr_rate, tnr1=0.72781, tnr0=0.72781),
#                                  Accuracy(), DIBinary(), 5))
#     algos1.append(FairGridSearch(ulr_eq_opp(s_as_input=True, use_bias=True, tpr0=tpr_rate,
#                                             tpr1=tpr_rate, tnr1=0.739723, tnr0=0.739723),
#                                  Accuracy(), DIBinary(), 5))
#     algos1.append(FairGridSearch(ulr_eq_opp(s_as_input=False, use_bias=True, tpr0=tpr_rate,
#                                             tpr1=tpr_rate, tnr1=0.750231, tnr0=0.750231),
#                                  Accuracy(), DIBinary(), 5))

# for tnr0 in [0.75]:  # [0.6, 0.7, 0.8, 0.9, 1.0]:
#     for tnr1 in [0.75]:  # [0.6, 0.7, 0.8, 0.9, 1.0]:
# for tpr in [0.6, 0.7, 0.8, 0.9, 1.0]:
#     tnr_in_true_race = 0.71
#     tnr_in_false_race = 0.74
#     tnr_in_true_sex = 0.72
#     tnr_in_false_sex = 0.77
#     algos1 += [GPyTEqOdds(s_as_input=True, tnr0=tnr_in_true_race, tnr1=tnr_in_true_race,
#                           tpr0=tpr, tpr1=tpr)]
#     algos1 += [GPyTEqOdds(s_as_input=False, tnr0=tnr_in_false_race, tnr1=tnr_in_false_race,
#                           tpr0=tpr, tpr1=tpr)]
#     algos1 += [GPyTEqOdds(s_as_input=True, tnr0=tnr_in_true_sex, tnr1=tnr_in_true_sex,
#                           tpr0=tpr, tpr1=tpr)]
#     algos1 += [GPyTEqOdds(s_as_input=False, tnr0=tnr_in_false_sex, tnr1=tnr_in_false_sex,
#                           tpr0=tpr, tpr1=tpr)]

# for pr in [.1, .2, .3, .4, .5]:
#     algos1 += [GPyTDemPar(s_as_input=True, target_acceptance=pr)]
#     algos1 += [GPyTDemPar(s_as_input=False, target_acceptance=pr)]

# for npv in [.1]:
#     algos1 += [GPyTCal(s_as_input=True, ppv0=0.75, ppv1=0.73, npv0=npv, npv1=npv)]
#     algos1 += [GPyTCal(s_as_input=False, ppv0=0.75, ppv1=0.73, npv0=npv, npv1=npv)]


ALGORITHMS = [
    # FairGridSearch(ulr_dem_par(s_as_input=True, use_bias=True), Accuracy(), DIBinary(), 5),
    # FairGridSearch(ulr_dem_par(s_as_input=False, use_bias=True), Accuracy(), DIBinary(), 5),
    # FairGridSearch(ulr(s_as_input=True, use_bias=True), Accuracy(), TPRNormDiff(), 5),
    # FairGridSearch(ulr(s_as_input=False, use_bias=True), Accuracy(), TPRNormDiff(), 5),
    # FairGridSearch(ulr(s_as_input=True, use_bias=True), Accuracy(), DIBinary(), 5),
    # FairGridSearch(ulr(s_as_input=False, use_bias=True), Accuracy(), DIBinary(), 5),
    # ParamGridSearch(ulr_dem_par(s_as_input=True, use_bias=False), Accuracy()),
    # ParamGridSearch(ulr_dem_par(s_as_input=False, use_bias=True), Accuracy()),
    # ParamGridSearch(ulr_dem_par(s_as_input=False, use_bias=False), Accuracy())

    # ParamGridSearch(ULREqOpp(s_as_input=True, use_bias=True, tnr0=.7, tnr1=.7, tpr0=.7, tpr1=.7), Accuracy()),
    # ParamGridSearch(ULREqOpp(s_as_input=True, use_bias=False, tnr0=.7, tnr1=.7, tpr0=.7, tpr1=.7), Accuracy()),
    # ParamGridSearch(ULREqOpp(s_as_input=False, use_bias=True, tnr0=.7, tnr1=.7, tpr0=.7, tpr1=.7), Accuracy()),
    # ParamGridSearch(ULREqOpp(s_as_input=False, use_bias=False, tnr0=.7, tnr1=.7, tpr0=.7, tpr1=.7), Accuracy())

    # -------- use l2_factor=0.00035 for the adult dataset ---------
    # ulr(s_as_input=True, l2_factor=0.0001, use_bias=True),
    # ulr(s_as_input=False, l2_factor=0.0001, use_bias=True),
    # ulr_dem_par(s_as_input=True, l2_factor=0.0001, use_bias=True),
    # ulr_dem_par(s_as_input=False, l2_factor=0.0001, use_bias=True),
    # -------- use l2_factor=0.0024 for the propublica dataset ---------
    # ulr(s_as_input=True, l2_factor=0.0024, use_bias=True),
    # ulr(s_as_input=False, l2_factor=0.0024, use_bias=True),
    # ulr_eq_opp(s_as_input=True, l2_factor=0.0024, use_bias=True, tnr0=.7, tnr1=.7, tpr0=.7, tpr1=.7),
    # ulr_eq_opp(s_as_input=True, l2_factor=0.0024, use_bias=False, tnr0=.7, tnr1=.7, tpr0=.7, tpr1=.7),
    # ulr_eq_opp(s_as_input=False, l2_factor=0.0024, use_bias=True, tnr0=.7, tnr1=.7, tpr0=.7, tpr1=.7),
    # ulr_eq_opp(s_as_input=False, l2_factor=0.0024, use_bias=False, tnr0=.7, tnr1=.7, tpr0=.7, tpr1=.7),

    GPyT(s_as_input=True),
    GPyT(s_as_input=False),
    GPyTDemPar(s_as_input=True),
    GPyTDemPar(s_as_input=False),
    GPyTDemPar(s_as_input=True, target_mode=UGPDemPar.MIN),
    GPyTDemPar(s_as_input=False, target_mode=UGPDemPar.MIN),
    GPyTDemPar(s_as_input=True, target_mode=UGPDemPar.MAX),
    GPyTDemPar(s_as_input=False, target_mode=UGPDemPar.MAX),
    GPyTEqOdds(s_as_input=True),
    GPyTEqOdds(s_as_input=False),
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

ALGORITHMS += algos1

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
