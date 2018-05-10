"""Code for calling UniversalGP"""
from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import call
import json
import numpy as np

from ..Algorithm import Algorithm

UGP_PATH = "/home/ubuntu/code/UniversalGP/gaussian_process.py"  # TODO: find a better way to specify the path
USE_EAGER = False
EPOCHS = 500
MAX_TRAIN_STEPS = 20000
BATCH_SIZE = 50


class UGP(Algorithm):
    """
    This class calls the UniversalGP code
    """

    def __init__(self, s_as_input=True):
        super().__init__()
        self.counter = 0
        self.s_as_input = s_as_input
        self.name = f"UGP_in_{s_as_input}"

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
            self.run_ugp(_flags(parameters, tmpdir, self.s_as_input, model_name, raw_data['ytrain'].shape[0]))

            # Read the results from the numpy file 'predictions.npz'
            output = np.load(tmp_path / Path(model_name) / Path("predictions.npz"))
            pred_mean = output['pred_mean']

        # Convert the result to the expected format
        return label_converter((pred_mean > 0.5).astype(raw_data['ytest'].dtype)[:, 0]), []

    @staticmethod
    def run_ugp(flags):
        """Run UniversalGP as a separte process"""
        cmd = f"python {UGP_PATH} "
        for key, value in flags.items():
            if isinstance(value, str):
                cmd += f" --{key}='{value}'"
            else:
                cmd += f" --{key}={value}"
        call(cmd, shell=True)  # run `cmd`

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


class UGPDemPar(UGP):
    """GP algorithm which enforces demographic parity"""
    def __init__(self, s_as_input=True, target_acceptance=None):
        super().__init__(s_as_input=s_as_input)
        if target_acceptance is None:
            self.name = f"UGP_dem_par_in_{s_as_input}"
        else:
            self.name = f"UGP_dem_par_in_{s_as_input}_tar_{target_acceptance}"
        self.target_acceptance = target_acceptance

    def _additional_parameters(self, raw_data):
        biased_acceptance1, biased_acceptance2 = compute_bias(raw_data['ytrain'], raw_data['strain'])

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


class UGPEqOpp(UGP):
    """GP algorithm which enforces equality of opportunity"""
    def __init__(self, s_as_input=True):
        super().__init__(s_as_input=s_as_input)
        self.name = f"UGP_eq_opp_in_{s_as_input}"

    def _additional_parameters(self, raw_data):
        biased_acceptance1, biased_acceptance2 = compute_bias(raw_data['ytrain'], raw_data['strain'])

        return dict(
            inf='VariationalYbarEqOdds',
            p_ybary0_s0=1.0,
            p_ybary0_s1=1.0,
            p_ybary1_s0=1.0,
            p_ybary1_s1=1.0,
            biased_acceptance1=biased_acceptance1,
            biased_acceptance2=biased_acceptance2,
        )

    def run(self, *data):
        self.counter += 1
        raw_data, label_converter = _prepare_data(*data)

        parameters = self._additional_parameters(raw_data)

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_name = "local"  # f"run{self.counter}_s_as_input_{self.s_as_input}"

            # Split the training data into train and dev and save it to `data.npz`
            train_dev_data = _split_train_dev(raw_data['xtrain'], raw_data['ytrain'], raw_data['strain'])
            np.savez(tmp_path / Path("data.npz"), **train_dev_data)

            # First run
            flags = _flags(parameters, tmpdir, self.s_as_input, model_name, raw_data['ytrain'].shape[0])
            self.run_ugp(flags)

            # Read the results from the numpy file 'predictions.npz'
            prediction_on_train = np.load(tmp_path / Path(model_name) / Path("predictions.npz"))
            preds = (prediction_on_train['pred_mean'] > 0.5).astype(int)
            odds = _compute_odds(train_dev_data['ytest'], preds, train_dev_data['stest'])

            # Enforce equality of opportunity
            opportunity = min(odds['p_ybary1_s0'], odds['p_ybary1_s1'])
            odds['p_ybary1_s0'] = opportunity
            odds['p_ybary1_s1'] = opportunity

            # Save with real test data
            np.savez(tmp_path / Path("data.npz"), **raw_data)

            # Second run
            flags.update({'train_steps': 2 * flags['train_steps'], **odds})
            self.run_ugp(flags)

            # Read the results from the numpy file 'predictions.npz'
            output = np.load(tmp_path / Path(model_name) / Path("predictions.npz"))
            pred_mean = output['pred_mean']

        # Convert the result to the expected format
        return label_converter((pred_mean > 0.5).astype(raw_data['ytest'].dtype)[:, 0]), []


def _prepare_data(train_df, test_df, class_attr, positive_class_val, sensitive_attrs, single_sensitive,
                  privileged_vals, params):
    # Separate data
    sensitive = [df[single_sensitive].values[:, np.newaxis] for df in [train_df, test_df]]
    label = [df[class_attr].values[:, np.newaxis] for df in [train_df, test_df]]
    nosensitive = [df.drop(columns=sensitive_attrs).drop(columns=class_attr).values for df in [train_df, test_df]]

    # Check sensitive attributes
    assert list(np.unique(sensitive[0])) == [0, 1] or list(np.unique(sensitive[0])) == [0., 1.]

    # Check labels
    label, label_converter = fix_labels(label, positive_class_val)
    return dict(xtrain=nosensitive[0], xtest=nosensitive[1], ytrain=label[0], ytest=label[1], strain=sensitive[0],
                stest=sensitive[1]), label_converter


def compute_bias(labels, sensitive):
    """Compute the bias in the labels with respect to the sensitive attributes"""
    rate_y1_s0 = np.sum(labels[sensitive == 0] == 1) / np.sum(sensitive == 0)
    rate_y1_s1 = np.sum(labels[sensitive == 1] == 1) / np.sum(sensitive == 1)
    return rate_y1_s0, rate_y1_s1


def _compute_odds(labels, predictions, sensitive):
    """Compute the bias in the predictions with respect to the sensitive attributes and the labels"""
    return dict(
        p_ybary0_s0=np.mean(predictions[np.logical_and(labels == 0, sensitive == 0)] == 0),
        p_ybary1_s0=np.mean(predictions[np.logical_and(labels == 1, sensitive == 0)] == 1),
        p_ybary0_s1=np.mean(predictions[np.logical_and(labels == 0, sensitive == 1)] == 0),
        p_ybary1_s1=np.mean(predictions[np.logical_and(labels == 1, sensitive == 1)] == 1),
    )


def fix_labels(labels, positive_class_val):
    """Make sure that labels are either 0 or 1

    Args"
        labels: the labels as a list of numpy arrays
        positive_class_val: the value that corresponds to a "positive" predictions

    Returns:
        the fixed labels and a function to convert the fixed labels back to the original format
    """
    label_values = list(np.unique(labels[0]))
    if label_values == [0, 1] and positive_class_val == 1:

        def _do_nothing(inp):
            return inp
        return labels, _do_nothing
    elif label_values == [1, 2] and positive_class_val == 1:

        def _converter(label):
            return 2 - label
        return [2 - y for y in labels], _converter
    raise ValueError("Labels have unknown structure")


def _split_train_dev(inputs, labels, sensitive):
    n = inputs.shape[0]
    idx_s0_y0 = np.where((sensitive == 0) & (labels == 0))[0]
    idx_s0_y1 = np.where((sensitive == 0) & (labels == 1))[0]
    idx_s1_y0 = np.where((sensitive == 1) & (labels == 0))[0]
    idx_s1_y1 = np.where((sensitive == 1) & (labels == 1))[0]

    train_fraction = []
    test_fraction = []
    for a in [idx_s0_y0, idx_s0_y1, idx_s1_y0, idx_s1_y1]:
        np.random.shuffle(a)

        split_idx = int(len(a) * 0.5) + 1  # make sure the train part is at least half
        train_fraction_a = a[:split_idx]
        test_fraction_a = a[split_idx:]
        train_fraction += list(train_fraction_a)
        test_fraction += list(test_fraction_a)
    xtrain, ytrain, strain = inputs[train_fraction], labels[train_fraction], sensitive[train_fraction]
    # ensure that the train set has exactly the same size as the given set (otherwise inducing inputs has wrong shape)
    return dict(xtrain=np.concatenate((xtrain, xtrain))[:n], ytrain=np.concatenate((ytrain, ytrain))[:n],
                strain=np.concatenate((strain, strain))[:n], xtest=inputs[test_fraction], ytest=labels[test_fraction],
                stest=sensitive[test_fraction])


def _flags(additional, save_dir, s_as_input, model_name, num_train):
    return {**dict(
        tf_mode='eager' if USE_EAGER else 'graph',
        data='sensitive_from_numpy',
        dataset_dir=save_dir,
        cov='SquaredExponential',
        lr=0.005,
        model_name=model_name,
        batch_size=BATCH_SIZE,
        train_steps=min(MAX_TRAIN_STEPS, num_train * EPOCHS // BATCH_SIZE),
        eval_epochs=100000,
        summary_steps=100000,
        chkpnt_steps=100000,
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
