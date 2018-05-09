import pandas as pd
from data.objects.Data import Data

class Ricci(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'ricci'
        # Class attribute will not be created until data_specific_processing is run.
        self.class_attr = 'class'
        self.positive_class_val = 1
        self.sensitive_attrs = ['race']
        self.privileged_class_names = ['W']
        self.categorical_features = ['position']
        self.features_to_keep = ['position', 'oral', 'written', 'race', 'combine']
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        dataframe['class'] = dataframe.apply(passing_grade, axis=1)
        return dataframe

    def handle_missing_data(self, dataframe):
        return dataframe

def passing_grade(row):
    """
    A passing grade in the Ricci data is defined as any grade above a 70 in the combined
    oral and written score.  (See Miao 2010.)
    """
    if row['Combine'] >= 70.0:
        return 1
    else:
        return 0
