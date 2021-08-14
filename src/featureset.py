# featureset.py
import pandas as pd
from util import Util 

class Featureset:
    name = '' # designation
    config = None # Config file for statics
    df = None # pandas dataframe to store the features (and indexed with the data from the sets)
    data_df = None # dataframe to get audio paths


    def __init__(self, name, config, data_df):
        self.name = name
        self.config = config
        self.data_df = data_df
        self.util = Util(config)

    def extract(self):
        pass

    def filter(self):
        # use only the features that are indexed in the target dataframes
        print(self.df.index[0])
        print(self.data_df.index[0])
        self.df = self.df[self.df.index.isin(self.data_df.index)]
