"""Code for calling Fair logistic regression"""

from .UGPAlgorithm import UGP, UGPDemPar, UGPEqOpp

# TODO: find a better way to specify the path
UGP_PATH = "/home/ubuntu/code/UniversalGP/gaussian_process.py"
BATCH_SIZE = 32
EPOCHS = 10


class ULR(UGP):
    """Logistic regression"""
    basename = "ULR"
    inference_method = "LogReg"

    def __init__(self, l2_factor=0.1, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.l2_factor = l2_factor
        self.use_bias = use_bias

    def _additional_parameters(self, raw_data):
        return _log_reg_params(super(), self, raw_data)

    @staticmethod
    def get_param_info():
        return dict(l2_factor=[], use_bias=[True, False])

    def get_default_params(self):
        return dict(l2_factor=self.l2_factor, use_bias=self.use_bias)


class ULRDemPar(UGPDemPar):
    """Logistic regression algorithm which enforces demographic parity"""
    basename = "ULR"
    inference_method = "FairLogReg"

    def __init__(self, l2_factor=0.1, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.l2_factor = l2_factor
        self.use_bias = use_bias

    def _additional_parameters(self, raw_data):
        return _log_reg_params(super(), self, raw_data)

    @staticmethod
    def get_param_info():
        return dict(l2_factor=[], use_bias=[True, False])

    def get_default_params(self):
        return dict(l2_factor=self.l2_factor, use_bias=self.use_bias)


class ULREqOpp(UGPEqOpp):
    """Logistic regression algorithm which enforces equality of opportunity"""
    basename = "ULR"
    inference_method = "EqOddsLogReg"

    def __init__(self, l2_factor=0.1, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.l2_factor = l2_factor
        self.use_bias = use_bias

    def _additional_parameters(self, raw_data):
        return _log_reg_params(super(), self, raw_data)

    @staticmethod
    def get_param_info():
        return dict(l2_factor=[], use_bias=[True, False])

    def get_default_params(self):
        return dict(l2_factor=self.l2_factor, use_bias=self.use_bias)


def _log_reg_params(base_obj, obj, raw_data):
    """Get the base parameters from `base_obj` and add the parameters for logistic regression"""
    params = base_obj._additional_parameters(raw_data)
    params.update(dict(
        batch_size=BATCH_SIZE,
        train_steps=(raw_data['ytrain'].shape[0] * EPOCHS) // BATCH_SIZE,
        dataset_standardize=True,
        use_bias=obj.use_bias,
        lr_l2_kernel_factor=obj.l2_factor,
        lr_l2_bias_factor=0.0,
        inf=obj.inference_method,
    ))
    return params
