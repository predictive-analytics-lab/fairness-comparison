from sklearn.tree import DecisionTreeClassifier as SKLearn_DT
from algorithms.baseline.Generic import Generic

class DecisionTree(Generic):
    def __init__(self, min_samples_split=0.5, min_samples_leaf=0.2):
        Generic.__init__(self)
        self.classifier = SKLearn_DT(min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf)
        # self.classifier = SKLearn_DT()
        self.name = f"DT_sp_{min_samples_split}_lf_{min_samples_leaf}"
