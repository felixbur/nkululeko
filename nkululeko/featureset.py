# featureset.py
import pandas as pd
from nkululeko.util import Util 
import nkululeko.glob_conf as glob_conf
import ast

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
        try: 
            # use only some features
            selected_features = ast.literal_eval(glob_conf.config['FEATS']['features'])
            self.util.debug(f'trying to select features: {selected_features}')
            sel_feats_df = pd.DataFrame()
            hit = False
            for feat in selected_features:
                try:
                    sel_feats_df[feat] = self.df[feat]
                    hit = True
                except KeyError:
                    pass
            if hit:
                self.df = sel_feats_df
                self.util.debug(f'new feats shape after selecting features: {self.df.shape}')
        except KeyError:
            pass

