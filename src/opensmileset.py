# opensmileset.py
from featureset import Featureset
import opensmile
import os
import pandas as pd
from util import Util 

class Opensmileset(Featureset):

    def extract(self):
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        if not os.path.isfile(storage):
            print('extracting openSmile features, this might take a while...')
            smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,)
            self.df = smile.process_files(self.data_df.index)
            self.df.to_pickle(storage)
        else:
            self.df = pd.read_pickle(storage)
        # drop the multiindex
        self.df.index = self.df.index.droplevel(1)
        self.df.index = self.df.index.droplevel(1)
