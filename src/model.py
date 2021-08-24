# model.py
from util import Util 
import glob_conf
from sklearn.utils import class_weight

class Model:
    """Generic model class"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test
        self.util = Util()
        target = glob_conf.config['DATA']['target']
        self.classes_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=df_train[target]
        )
