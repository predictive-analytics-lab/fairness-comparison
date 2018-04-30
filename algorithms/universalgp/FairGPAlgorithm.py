"""Code for calling UniversalGP"""
from collections import namedtuple

import numpy as np
import tensorflow as tf

import universalgp as ugp
from universalgp.datasets.definition import Dataset, to_tf_dataset_fn

from ..Algorithm import Algorithm

USE_EAGER = False
DO_FAIR = True
MAX_NUM_INDUCING = 500


class FairGPAlgorithm(Algorithm):
    """
    This class calls the UniversalGP code
    """
    name = "FairGP"

    def __init__(self, s_as_input=True):
        Algorithm.__init__(self)
        self.counter = 0
        self.s_as_input = s_as_input

    def run(self, *data):
        """
        Runs the algorithm and returns the predicted classifications on the test set.  The given train and test data
        still contains the sensitive_attrs.  This run of the algorithm should focus on the single given sensitive
        attribute.

        Be sure that the returned predicted classifications are of the same type as the class attribute in the given
        test_df.  If this is not the case, some metric analyses may fail to appropriately compare the returned
        predictions to their desired values.

        Args:
            train_df: Pandas datafram with the training data
            test_df: Pandas datafram with the test data
            class_attr: string that names the column with the label
            positive_class_val: the value for the label which is considered the positive class (usually '1')
            sensitive_attrs: list of all available sensitive attributes (all but one should be ignored)
            single_sensitive: string that names the sensitive attribute that is considered in this run
            privileged_vals: the groups that are considered privileged (usually '1')
            params: a dictionary mapping from algorithm-specific parameter names to the desired values.
                If the implementation of run uses different values, these should be modified in the params
                dictionary as a way of returning the used values to the caller.
        """
        self.counter += 1
        (train, test), label_converter, params = _prepare_data(*data)
        num_train = train.x.shape[0]
        num_inducing = min(num_train, MAX_NUM_INDUCING)

        if self.s_as_input:
            inducing_inputs = np.concatenate((train.x[::num_train // num_inducing],
                                              train.s[::num_train // num_inducing]), -1)
        else:
            inducing_inputs = train.x[::num_train // num_inducing]

        dataset = Dataset(
            train_fn=to_tf_dataset_fn(train.x, train.y, train.s),
            test_fn=to_tf_dataset_fn(test.x, test.y, test.s),
            input_dim=train.x.shape[1] + 1 if self.s_as_input else train.x.shape[1],
            # xtrain=train.x,
            # ytrain=train.y,
            # strain=train.s,
            # xtest=test.x,
            # ytest=test.y,
            # stest=test.s,
            num_train=num_train,
            inducing_inputs=inducing_inputs,
            output_dim=train.y.shape[1],
            lik="LikelihoodLogistic",
            metric="logistic_accuracy,pred_rate_y1_s0,pred_rate_y1_s1,base_rate_y1_s0,base_rate_y1_s1",
        )

        if USE_EAGER:
            try:
                tf.enable_eager_execution()
            except ValueError:
                pass
            train_func = ugp.train_eager.train_gp
        else:
            tf.logging.set_verbosity(tf.logging.INFO)
            train_func = ugp.train_graph.train_gp

        biased_acceptance1, biased_acceptance2 = _compute_bias(train.y, train.s)

        gp = train_func(dataset, dict(
            inf='VariationalYbar' if DO_FAIR else 'Variational',
            cov='SquaredExponential',
            lr=0.005,
            loo_steps=None,
            model_name=f"run{self.counter}_s_as_input_{self.s_as_input}",
            batch_size=50,
            train_steps=1000,
            eval_epochs=10000,
            summary_steps=100,
            chkpnt_steps=100,
            save_dir=None,  # "/home/ubuntu/out2/",
            plot=None,
            logging_steps=100,
            gpus='0',
            save_preds=False,
            num_components=1,
            num_samples=1000,
            diag_post=False,
            optimize_inducing=True,
            use_loo=False,
            length_scale=1.0,
            sf=1.0,
            iso=False,
            num_samples_pred=2000,
            target_rate1=.5 * (biased_acceptance1 + biased_acceptance2),
            target_rate2=.5 * (biased_acceptance1 + biased_acceptance2),
            biased_acceptance1=biased_acceptance1,
            biased_acceptance2=biased_acceptance2,
            s_as_input=self.s_as_input,
            probs_from_flipped=False,
        ))

        if USE_EAGER:
            pred_mean, _ = gp.predict({'input': test.x})
            pred_mean = pred_mean.numpy()
        else:
            predictions_gen = gp.predict(lambda: to_tf_dataset_fn(test.x, test.y, test.s)().batch(50))
            pred_mean = []
            # pred_var = []
            for prediction in predictions_gen:
                pred_mean.append(prediction['mean'])
                # pred_var.append(prediction['var'])
            pred_mean = np.stack(pred_mean)
            # pred_var = np.stack(pred_var)
        return label_converter((pred_mean > 0.5).astype(test.y.dtype)[:, 0]), []

    @staticmethod
    def get_param_info():
        """
        Returns a dictionary mapping algorithm parameter names to a list of parameter values to be explored. This
        function should only be implemented if the algorithm has specific parameters that should be tuned, e.g., for
        trading off between fairness and accuracy.
        """
        return dict(s_as_input=[True, False])

    @staticmethod
    def get_supported_data_types():

        return set(["numerical-binsensitive"])

    def get_name(self):
        """
        Returns the name for the algorithm. This must be a unique name, so it is suggested that this name is simply
        <firstauthor>. If there are mutliple algorithms by the same author(s), a suggested modification is
        <firstauthor-algname>. This name will appear in the resulting CSVs and graphs created when performing
        benchmarks and analysis.
        """
        return self.name

    def get_default_params(self):
        """
        Returns a dictionary mapping from parameter names to default values that should be used with the algorithm. If
        not implemented by a specific algorithm, this returns the empty dictionary.
        """
        return dict(s_as_input=self.s_as_input)


def _prepare_data(train_df, test_df, class_attr, positive_class_val, sensitive_attrs, single_sensitive,
                  privileged_vals, params):
    Data = namedtuple('Data', ['x', 'y', 's'])

    # Separate data
    sensitive = [df[single_sensitive].values[:, np.newaxis] for df in [train_df, test_df]]
    label = [df[class_attr].values[:, np.newaxis] for df in [train_df, test_df]]
    nosensitive = [df.drop(columns=sensitive_attrs).drop(columns=class_attr).values for df in [train_df, test_df]]

    # Check sensitive attributes
    assert list(np.unique(sensitive[0])) == [0, 1] or list(np.unique(sensitive[0])) == [0., 1.]

    # Normalize input
    input_normalizer = _get_normalizer(nosensitive[0])  # the normalizer must be based only on the training data
    nosensitive = [input_normalizer(x) for x in nosensitive]

    # Check labels
    label, label_converter = _fix_labels(label, positive_class_val)
    return [Data(x=x, y=y, s=s) for x, y, s in zip(nosensitive, label, sensitive)], label_converter, params


def _compute_bias(labels, sensitive):
    rate_y1_s0 = np.sum(labels[sensitive == 0] == 1) / np.sum(sensitive == 0)
    rate_y1_s1 = np.sum(labels[sensitive == 1] == 1) / np.sum(sensitive == 1)
    return rate_y1_s0, rate_y1_s1


def _get_normalizer(base):
    if base.min() == 0 and base.max() > 10:
        max_per_feature = np.amax(base, axis=0)

        def normalizer(unnormalized):
            return unnormalized / max_per_feature
        return normalizer

    def do_nothing(inp):
        return inp
    return do_nothing


def _fix_labels(labels, positive_class_val):
    label_values = list(np.unique(labels[0]))
    if label_values == [0, 1] and positive_class_val == 1:

        def do_nothing(inp):
            return inp
        return labels, do_nothing
    elif label_values == [1, 2] and positive_class_val == 1:

        def converter(label):
            return 2 - label
        return [2 - y for y in labels], converter
    else:
        raise ValueError("Labels have unknown structure")
