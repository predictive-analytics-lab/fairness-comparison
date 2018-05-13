from data.objects.Data import Data
import numpy as np
import pandas as pd

##############################################################################


class FlippedLabels(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'flipped-labels'
        self.class_attr = 'label'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sensitive-attr']
        self.privileged_class_names = ['1']
        self.categorical_features = []
        self.features_to_keep = ['x1', 'x2', 'sensitive-attr', 'label']
        self.missing_val_indicators = ['?']

    def load_raw_dataset(self):
        pass
