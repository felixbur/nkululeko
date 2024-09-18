# featureset.py
import ast

import pandas as pd

import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


class Featureset:
    name = ""  # designation
    df = None  # pandas dataframe to store the features
    # (and indexed with the data from the sets)
    data_df = None  # dataframe to get audio paths

    def __init__(self, name, data_df, feats_type):
        self.name = name
        self.data_df = data_df
        self.util = Util("featureset")
        self.feats_type = feats_type
        self.n_jobs = int(self.util.config_val("MODEL", "n_jobs", "8"))

    def extract(self):
        pass

    def filter(self):
        # use only the features that are indexed in the target dataframes
        self.df = self.df[self.df.index.isin(self.data_df.index)]
        try:
            # use only some features
            selected_features = ast.literal_eval(glob_conf.config["FEATS"]["features"])
            self.util.debug(f"selecting features: {selected_features}")
            sel_feats_df = pd.DataFrame()
            hit = False
            for feat in selected_features:
                try:
                    sel_feats_df[feat] = self.df[feat]
                    hit = True
                except KeyError:
                    self.util.warn(f"non existent feature in {self.feats_type}: {feat}")
                    pass
            if hit:
                self.df = sel_feats_df
                self.util.debug(
                    f"new feats shape after selecting features for {self.feats_type}: {self.df.shape}"
                )
        except KeyError:
            pass
