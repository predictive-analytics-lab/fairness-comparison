"""Code for calling Fair logistic regression"""

import numpy as np
from .UGPAlgorithm import UGP, UGPDemPar, UGPEqOpp

# TODO: find a better way to specify the path
UGP_PATH = "/home/ubuntu/code/UniversalGP/gaussian_process.py"
BATCH_SIZE = 32
EPOCHS = 10
FACTOR_SET = np.power(10, np.linspace(-3, -1, 3))  # [0.1, 0.01, 0.001]


class ULRBase:
    """Base class for logistic regression

    This class cannot be used on its own. This class contains code that the classes `ULR`,
    `ULRDemPar` and `ULREqOpp` share. It is meant to be used with multiple inheritance.

    The `super_class` has to be the other class in the multiple inheritance.
    """
    basename = "ULR"

    def __init__(self, super_class, inference_method, l2_factor, use_bias, **kwargs):
        super_class.__init__(self, **kwargs)  # this only makes sense with multiple inheritance
        self.super_class = super_class
        self.inference_method = inference_method
        self.default_l2_factor = l2_factor
        self.use_bias = use_bias
        if use_bias:
            self.name += "_use_bias"

    def _additional_parameters(self, raw_data):
        """Get the base parameters from `base_obj` and add the parameters for logistic regression"""
        params = self.super_class._additional_parameters(self, raw_data)
        params.update(dict(
            batch_size=BATCH_SIZE,
            train_steps=(raw_data['ytrain'].shape[0] * EPOCHS) // BATCH_SIZE,
            dataset_standardize=True,
            use_bias=self.use_bias,
            lr_l2_kernel_factor=self.l2_factor,
            lr_l2_bias_factor=0.0,
            inf=self.inference_method,
        ))
        return params

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        self.l2_factor = params.get('l2_factor', self.default_l2_factor)
        return self.super_class.run(self, train_df, test_df, class_attr, positive_class_val,
                                    sensitive_attrs, single_sensitive, privileged_vals, params)

    @staticmethod
    def get_param_info():
        return dict(l2_factor=FACTOR_SET, use_bias=[])

    def get_default_params(self):
        return dict(l2_factor=self.default_l2_factor, use_bias=self.use_bias)


class ULR(ULRBase, UGP):
    """Logistic regression"""
    pass


class ULRDemPar(ULRBase, UGPDemPar):
    """Logistic regression algorithm which enforces demographic parity"""
    pass


class ULREqOpp(ULRBase, UGPEqOpp):
    """Logistic regression algorithm which enforces equality of opportunity"""
    pass


# Helper functions for constructing the ULR objects:

def ulr(l2_factor=0.1, use_bias=True, **kwargs):
    """Logistic regression"""
    return ULR(super_class=UGP, inference_method="LogReg", l2_factor=l2_factor,
               use_bias=use_bias, **kwargs)


def ulr_dem_par(l2_factor=0.1, use_bias=True, **kwargs):
    """Logistic regression algorithm which enforces demographic parity"""
    return ULRDemPar(super_class=UGPDemPar, inference_method="FairLogReg", l2_factor=l2_factor,
                     use_bias=use_bias, **kwargs)


def ulr_eq_opp(l2_factor=0.1, use_bias=True, **kwargs):
    """Logistic regression algorithm which enforces equality of opportunity"""
    return ULREqOpp(super_class=UGPEqOpp, inference_method="EqOddsLogReg", l2_factor=l2_factor,
                    use_bias=use_bias, **kwargs)
