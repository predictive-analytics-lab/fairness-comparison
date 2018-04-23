"""Class for algo-fairness framework"""
from algorithms.Algorithm import Algorithm
import tensorflow.contrib.eager as tfe
import numpy as np
import pandas as pd
import tensorflow as tf
from algorithms.universalgp.UniversalGPmaster.datasets.definition import Dataset, to_tf_dataset_fn
import algorithms.universalgp.UniversalGPmaster.universalgp as ugp


class UniversalGPAlgorithm(Algorithm):
    """
    This class calls the UniversalGP code
    """

    def __init__(self):
        Algorithm.__init__(self)
        self.name = 'UniversalGP'

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        """
        Runs the algorithm and returns the predicted classifications on the test set.  The given
        train and test data still contains the sensitive_attrs.  This run of the algorithm
        should focus on the single given sensitive attribute.

        Be sure that the returned predicted classifications are of the same type as the class
        attribute in the given test_df.  If this is not the case, some metric analyses may fail to
        appropriately compare the returned predictions to their desired values.

        Args:
            train_df: Pandas datafram with the training data
            test_df: Pandas datafram with the test data
            class_attr:
            positive_class_val:
            sensitive_attrs:
            single_sensitive:
            privileged_vals:
            params: a dictionary mapping from algorithm-specific parameter names to the desired values.
                If the implementation of run uses different values, these should be modified in the params
                dictionary as a way of returning the used values to the caller.
        """
        # remove sensitive attributes from the training set
        train_sensitive = train_df[single_sensitive]
        train_target = train_df[class_attr]
        train_df_nosensitive = train_df.drop(columns=sensitive_attrs)
        train_df_nosensitive = train_df_nosensitive.drop(columns=class_attr)

        test_sensitive = test_df[single_sensitive]
        # test_sensitive = pd.DataFrame()
        # test_sensitive = pd.get_dummies(test_sensitive)
        test_target = test_df[class_attr]
        test_df_nosensitive = test_df.drop(columns=sensitive_attrs)
        test_df_nosensitive = test_df_nosensitive.drop(columns=class_attr)

        data_train = (train_df_nosensitive.values, train_sensitive.values.reshape(-1, 1),
                      train_target.values.reshape(-1, 1))
        data_test = (test_df_nosensitive.values, test_sensitive.values.reshape(-1, 1),
                     test_target.values.reshape(-1, 1))

        num_train = data_train[0].shape[0]
        x_size = data_train[0].shape[1]
        # s_size = data_train[1].shape[1]
        y_size = data_train[2].shape[1]

        dataset = Dataset(
            train_fn=to_tf_dataset_fn(data_train[0], data_train[2], data_train[1]),
            test_fn=to_tf_dataset_fn(data_test[0], data_test[2], data_test[1]),
            input_dim=x_size,
            xtrain=data_train[0],
            ytrain=data_train[2],
            strain=data_train[1],
            xtest=data_test[0],
            ytest=data_test[2],
            stest=data_test[1],
            num_train=num_train,
            inducing_inputs=data_train[0],
            output_dim=y_size,
            lik="LikelihoodLogistic",
            metric="logistic_accuracy"
        )
        use_eager = False
        if use_eager:
            try:
                tfe.enable_eager_execution()
            except ValueError:
                pass
            train_function = ugp.train_eager.train_gp
        else:
            tf.logging.set_verbosity(tf.logging.INFO)
            train_function = ugp.train_graph.train_gp
        gp = train_function(dataset, dict(
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
        if use_eager:
            pred_mean, pred_var = gp.predict({'input': data_test[0]})
            pred_mean = pred_mean.numpy()
        else:
            predictions_gen = gp.predict(lambda: to_tf_dataset_fn(data_test[0], data_test[2])().batch(50))
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
        Returns a dictionary mapping algorithm parameter names to a list of parameter values to
        be explored.  This function should only be implemented if the algorithm has specific
        parameters that should be tuned, e.g., for trading off between fairness and accuracy.
        """
        return {}

    def get_supported_data_types(self):

        return set(["numerical-binsensitive"])

    def get_name(self):
        """
        Returns the name for the algorithm.  This must be a unique name, so it is suggested that
        this name is simply <firstauthor>.  If there are mutliple algorithms by the same author(s), a
        suggested modification is <firstauthor-algname>.  This name will appear in the resulting
        CSVs and graphs created when performing benchmarks and analysis.
        """
        return self.name

    def get_default_params(self):
        """
        Returns a dictionary mapping from parameter names to default values that should be used with
        the algorithm.  If not implemented by a specific algorithm, this returns the empty
        dictionary.
        """
        return {}
