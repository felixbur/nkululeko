# featureset.py
import pandas as pd
from util import Util 
import glob_conf

class Featureset:
    name = '' # designation
    df = None # pandas dataframe to store the features (and indexed with the data from the sets)
    data_df = None # dataframe to get audio paths


    def __init__(self, name, data_df):
        self.name = name
        self.data_df = data_df
        self.util = Util()

    def extract(self):
        pass

    def filter(self):
        # use only the features that are indexed in the target dataframes
        self.df = self.df[self.df.index.isin(self.data_df.index)]
