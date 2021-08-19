# model.py
from util import Util 
import glob_conf

class Model:
    """Generic model class"""
    name = ''  # The model designation
    df_train, df_test, feats_train, feats_test = None, None, None, None # The data to train and evaluate the model

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes"""
        self.df_train, self.df_test, self.feats_train, self.feats_test = df_train, df_test, feats_train, feats_test
        self.util = Util()