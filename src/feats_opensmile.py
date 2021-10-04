# opensmileset.py
from featureset import Featureset
import opensmile
import os
import pandas as pd
from util import Util 
import glob_conf

class Opensmileset(Featureset):

    def __init__(self, name, data_df):
        super().__init__(name, data_df)
        self.feature_set = opensmile.FeatureSet.eGeMAPSv02

    def extract(self):
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        extract = self.util.config_val('DATA', 'needs_feature_extraction', False)
        if extract or not os.path.isfile(storage):
            self.util.debug('extracting openSmile features, this might take a while...')
            smile = opensmile.Smile(
            feature_set= self.feature_set,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=5,)
            print(self.data_df.head(1))
            if isinstance(self.data_df.index, pd.MultiIndex):
                self.df = smile.process_index(self.data_df.index)
            else:
                self.df = smile.process_files(self.data_df.index)
            print(self.df.shape)

            self.df.to_pickle(storage)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else:
            self.util.debug('reusing extracted OS features.')
            self.df = pd.read_pickle(storage)
        # drop the multiindex
        #self.df.index = self.df.index.droplevel(1)
        #self.df.index = self.df.index.droplevel(1)


    def extract_sample(self, signal, sr):
        smile = opensmile.Smile(
                feature_set=self.feature_set,
                feature_level=opensmile.FeatureLevel.Functionals,)
        feats = smile.process_signal(signal, sr)
        return feats