import pandas as pd
import numpy as np

# TAGS = ["original", "numerical", "numerical-binsensitive", "categorical-binsensitive"]
TAGS = ["numerical-binsensitive"]
TRAINING_PERCENT = 2.0 / 3.0
SEED = 1234  # the random seed


class ProcessedData():
    def __init__(self, data_obj):
        self.data = data_obj
        self.dfs = dict((k, pd.read_csv(self.data.get_filename(k)))
                        for k in TAGS)
        self.splits = dict((k, []) for k in TAGS)
        self.has_splits = False

    def get_processed_filename(self, tag):
        return self.data.get_filename(tag)

    def get_dataframe(self, tag):
        return self.dfs[tag]

    def create_train_test_splits(self, num, sensitive_attr):
        if self.has_splits:
            return self.splits

        class_attr = self.data.get_class_attribute()

        # get a local random state that is reproducible and doesn't affect other computations
        random = np.random.RandomState(SEED)

        for _ in range(num):
            # we first shuffle a list of indices so that each subprocessed data
            # is split consistently
            for k, df in self.dfs.items():
                if sensitive_attr in self.data.get_sensitive_attributes():
                    pos_val = self.data.get_positive_class_val(k)
                    idx_s0_y0 = np.where((df[sensitive_attr] == 0) & (df[class_attr] != pos_val))[0]
                    idx_s0_y1 = np.where((df[sensitive_attr] == 0) & (df[class_attr] == pos_val))[0]
                    idx_s1_y0 = np.where((df[sensitive_attr] == 1) & (df[class_attr] != pos_val))[0]
                    idx_s1_y1 = np.where((df[sensitive_attr] == 1) & (df[class_attr] == pos_val))[0]

                    train_fraction = []
                    test_fraction = []
                    for a in [idx_s0_y0, idx_s0_y1, idx_s1_y0, idx_s1_y1]:
                        random.shuffle(a)

                        split_idx = int(len(a) * TRAINING_PERCENT)
                        train_fraction_a = a[:split_idx]
                        test_fraction_a = a[split_idx:]
                        train_fraction += list(train_fraction_a)
                        test_fraction += list(test_fraction_a)
                elif sensitive_attr in self.data.get_sensitive_attributes_with_joint():
                    n = len(list(self.dfs.values())[0])

                    a = np.arange(n)
                    random.shuffle(a)

                    split_ix = int(n * TRAINING_PERCENT)
                    train_fraction = a[:split_ix]
                    test_fraction = a[split_ix:]
                else:
                    raise ValueError("Something is wrong")

                train = df.iloc[train_fraction]
                test = df.iloc[test_fraction]
                self.splits[k].append((train, test))

        self.has_splits = True
        return self.splits

    def get_sensitive_values(self, tag):
        """
        Returns a dictionary mapping sensitive attributes in the data to a list of all possible
        sensitive values that appear.
        """
        df = self.get_dataframe(tag)
        all_sens = self.data.get_sensitive_attributes_with_joint()
        sensdict = {}
        for sens in all_sens:
            sensdict[sens] = list(set(df[sens].values.tolist()))
        return sensdict
