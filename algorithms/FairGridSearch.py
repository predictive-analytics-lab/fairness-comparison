import numpy as np

from algorithms.ParamGridSearch import ParamGridSearch


class FairGridSearch(ParamGridSearch):
    """Grid search that takes into account a primary metric and a fairness metric"""
    def __init__(self, algorithm, metric, fairness_metric, topk):
        super().__init__(algorithm, metric)
        self.fairness_metric = fairness_metric
        self.topk = topk

    def find_best(self, all_predictions, train_df, test_df, class_attr, positive_class_val,
                  sensitive_attrs, single_sensitive, privileged_vals, params):
        if not all_predictions:
            raise Exception(
                "No run in the parameter grid search succeeded - failing run of algorithm")
        actual = test_df[class_attr]
        dict_sensitive = {sens: test_df[sens].values.tolist() for sens in sensitive_attrs}

        # find top K
        topk_indices = self.find_topk(all_predictions, positive_class_val, single_sensitive,
                                      privileged_vals, actual, dict_sensitive)

        best_val = None
        best = None
        best_name = None
        best_param_value = None
        for index in topk_indices:
            param_name, param_val, predictions = all_predictions[index]
            val = self.fairness_metric.calc(actual, predictions, dict_sensitive, single_sensitive,
                                            privileged_vals, positive_class_val)
            if best_val is None or self.fairness_metric.is_better_than(val, best_val):
                best = predictions
                best_name = param_name
                best_val = val
                best_param_value = param_val
        self.reset_params(best_name, best_param_value, params)
        return best

    def find_topk(self, all_predictions, positive_class_val, single_sensitive, privileged_vals,
                  actual, dict_sensitive):

        # compute values
        all_values = [
            self.metric.calc(actual, predictions, dict_sensitive, single_sensitive, privileged_vals,
                             positive_class_val)
            for _, _, predictions in all_predictions
        ]

        # start with random set of indices
        topk_indices = list(range(self.topk))

        # loop over the rest of the values
        for i, metric_value in enumerate(all_values[self.topk:]):
            # find minimum in topk set
            topk_values = [all_values[index] for index in topk_indices]
            min_value = min(topk_values)
            min_index = np.argmin(topk_values)

            # check if current value is greater than minimum of topk set
            if metric_value > min_value:
                topk_indices[min_index] = i + self.topk

        return topk_indices
