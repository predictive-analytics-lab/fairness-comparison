"""Code for calling UniversalGP"""
from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import call
import json
import numpy as np

from ..Algorithm import Algorithm

UGP_PATH = "/home/ubuntu/code/UniversalGP/gaussian_process.py"  # TODO: find a better way to specify the path
USE_EAGER = False


class GPAlgorithm(Algorithm):
    """
    This class calls the UniversalGP code
    """

    def __init__(self, s_as_input=True):
        super().__init__()
        self.counter = 0
        self.s_as_input = s_as_input
        self.name = f"GP_input_{s_as_input}"

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
        # Separate the data and make sure the labels are either 0 or 1
        raw_data, label_converter = _prepare_data(*data)

        # Set algorithm dependent parameters
        parameters = self._additional_parameters(raw_data)

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Save the data in a numpy file called 'data.npz'
            np.savez(tmp_path / Path("data.npz"), **raw_data)

            # Construct and execute command
            model_name = "local"  # f"run{self.counter}_s_as_input_{self.s_as_input}"
            cmd = f"python {UGP_PATH} "
            for key, value in _flags(parameters, tmpdir, self.s_as_input, model_name).items():
                if isinstance(value, str):
                    cmd += f" --{key}='{value}'"
                else:
                    cmd += f" --{key}={value}"
            call(cmd, shell=True)

            # Read the results from the numpy file 'predictions.npz'
            output = np.load(tmp_path / Path(model_name) / Path("predictions.npz"))
            pred_mean = output['pred_mean']

        # Convert the result to the expected format
        return label_converter((pred_mean > 0.5).astype(raw_data['ytest'].dtype)[:, 0]), []

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

    @staticmethod
    def _additional_parameters(_):
        return dict(
            inf='Variational',
        )

    def _save_in_json(self, save_path):
        """Save the settings in a JSON file called 'settings.json'"""
        with open(save_path / Path("settings.json"), 'w') as fp:
            json.dump(dict(s_as_input=self.s_as_input, counter=self.counter), fp, ensure_ascii=False, indent=2)


class FairGPAlgorithm(GPAlgorithm):
    """Fair GP algorithm"""
    def __init__(self, s_as_input=True, target_acceptance=None):
        super().__init__(s_as_input=s_as_input)
        if target_acceptance is None:
            self.name = f"FairGP_input_{s_as_input}"
        else:
            self.name = f"FairGP_in_{s_as_input}_tar_{target_acceptance}"
        self.target_acceptance = target_acceptance

    def _additional_parameters(self, raw_data):
        biased_acceptance1, biased_acceptance2 = _compute_bias(raw_data['ytrain'], raw_data['strain'])

        if self.target_acceptance is None:
            target_rate = .5 * (biased_acceptance1 + biased_acceptance2)
        else:
            target_rate = self.target_acceptance

        return dict(
            inf='VariationalYbar',
            target_rate1=target_rate,
            target_rate2=target_rate,
            biased_acceptance1=biased_acceptance1,
            biased_acceptance2=biased_acceptance2,
            probs_from_flipped=False,
        )


def _prepare_data(train_df, test_df, class_attr, positive_class_val, sensitive_attrs, single_sensitive,
                  privileged_vals, params):
    # Separate data
    sensitive = [df[single_sensitive].values[:, np.newaxis] for df in [train_df, test_df]]
    label = [df[class_attr].values[:, np.newaxis] for df in [train_df, test_df]]
    nosensitive = [df.drop(columns=sensitive_attrs).drop(columns=class_attr).values for df in [train_df, test_df]]

    # Check sensitive attributes
    assert list(np.unique(sensitive[0])) == [0, 1] or list(np.unique(sensitive[0])) == [0., 1.]

    # Check labels
    label, label_converter = _fix_labels(label, positive_class_val)
    return dict(xtrain=nosensitive[0], xtest=nosensitive[1], ytrain=label[0], ytest=label[1], strain=sensitive[0],
                stest=sensitive[1]), label_converter


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
    raise ValueError("Labels have unknown structure")


def _compute_bias(labels, sensitive):
    rate_y1_s0 = np.sum(labels[sensitive == 0] == 1) / np.sum(sensitive == 0)
    rate_y1_s1 = np.sum(labels[sensitive == 1] == 1) / np.sum(sensitive == 1)
    return rate_y1_s0, rate_y1_s1


def _flags(additional, save_dir, s_as_input, model_name):
    return {**dict(
        tf_mode='eager' if USE_EAGER else 'graph',
        data='sensitive_from_numpy',
        dataset_dir=save_dir,
        cov='SquaredExponential',
        lr=0.005,
        model_name=model_name,
        batch_size=50,
        train_steps=1000,
        eval_epochs=10000,
        summary_steps=5000,
        chkpnt_steps=5000,
        save_dir=save_dir,  # "/home/ubuntu/out2/",
        plot='',
        logging_steps=100,
        gpus='0',
        preds_path='predictions.npz',  # save the predictions into `predictions.npz`
        num_components=1,
        num_samples=1000,
        diag_post=False,
        optimize_inducing=True,
        use_loo=False,
        length_scale=1.0,
        sf=1.0,
        iso=False,
        num_samples_pred=2000,
        s_as_input=s_as_input,
    ), **additional}
