# opensmileset.py
from featureset import Featureset
import opensmile
import os
import pandas as pd

class Opensmileset(Featureset):

    def __init__(self, config, data_df):
        Featureset.__init__(self, config, data_df) 

    def extract(self):
        storage = f'store/{self.name}.pkl'
        if not os.path.isfile(storage):
            smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,)
            self.df = smile.process_files(self.data_df.index)
            self.df.to_pickle(storage)
        else:
            self.df = pd.read_pickle(storage)
