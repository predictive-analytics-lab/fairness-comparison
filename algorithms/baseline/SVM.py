from sklearn.svm import SVC as SKLearn_SVM
from algorithms.baseline.Generic import Generic


class SVM(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_SVM()
        self.name = "SVM"

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        self.classifier.set_params(C=params['C'])
        print(f"running with C = {params['C']}")
        return super().run(train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
                           single_sensitive, privileged_vals, params)

    def get_param_info(self):
        return {'C': [10., 1., .1, .01, .001]}

    def get_default_params(self):
        return {'C': 1.0}
