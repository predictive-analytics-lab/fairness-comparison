"""Code for calling UniversalGP"""
import numpy as np
import tensorflow as tf

import universalgp as ugp
from universalgp.datasets.definition import Dataset, to_tf_dataset_fn

from ..Algorithm import Algorithm

USE_EAGER = False


class UniversalGPAlgorithm(Algorithm):
    """
    This class calls the UniversalGP code
    """

    def __init__(self):
        Algorithm.__init__(self)
        self.name = 'UniversalGP'

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs, single_sensitive, privileged_vals,
            params):
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
        train_sensitive = train_df[single_sensitive].values[:, np.newaxis]
        train_label = train_df[class_attr].values[:, np.newaxis]
        train_df_nosensitive = train_df.drop(columns=sensitive_attrs).drop(columns=class_attr)
        train_nosensitive = train_df_nosensitive.values

        test_sensitive = test_df[single_sensitive].values[:, np.newaxis]
        test_label = test_df[class_attr].values[:, np.newaxis]
        test_df_nosensitive = test_df.drop(columns=sensitive_attrs).drop(columns=class_attr)
        test_nosensitive = test_df_nosensitive.values

        dataset = Dataset(
            train_fn=to_tf_dataset_fn(train_nosensitive, train_label, train_sensitive),
            test_fn=to_tf_dataset_fn(test_nosensitive, test_label, test_sensitive),
            input_dim=train_nosensitive.shape[1],
            # xtrain=train_nosensitive,
            # ytrain=train_label,
            # strain=train_sensitive,
            # xtest=test_nosensitive,
            # ytest=test_label,
            # stest=test_sensitive,
            num_train=train_nosensitive.shape[0],
            inducing_inputs=train_nosensitive,
            output_dim=train_label.shape[1],
            lik="LikelihoodLogistic",
            metric=""
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

        gp = train_func(dataset, dict(
            inf='Variational',
            cov='SquaredExponential',
            lr=0.005,
            loo_steps=None,
            model_name='local',
            batch_size=50,
            train_steps=50,
            eval_epochs=10000,
            summary_steps=100,
            chkpnt_steps=5000,
            save_dir=None,
            plot=None,
            logging_steps=1,
            gpus='0',
            save_preds=False,
            num_components=1,
            num_samples=100,
            diag_post=False,
            optimize_inducing=True,
            use_loo=False,
            length_scale=1.0,
            sf=1.0,
            iso=False,
            num_samples_pred=200
        ))

        if USE_EAGER:
            pred_mean, _ = gp.predict({'input': test_nosensitive})
            pred_mean = pred_mean.numpy()
        else:
            predictions_gen = gp.predict(lambda: to_tf_dataset_fn(test_nosensitive, test_label)().batch(50))
            pred_mean = []
            # pred_var = []
            for prediction in predictions_gen:
                pred_mean.append(prediction['mean'])
                # pred_var.append(prediction['var'])
            pred_mean = np.stack(pred_mean)
            # pred_var = np.stack(pred_var)
        return (pred_mean > 0.5).astype(np.float), []

    def get_param_info(self):
        """
        Returns a dictionary mapping algorithm parameter names to a list of parameter values to be explored. This
        function should only be implemented if the algorithm has specific parameters that should be tuned, e.g., for
        trading off between fairness and accuracy.
        """
        return {}

    def get_supported_data_types(self):

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
        return {}
