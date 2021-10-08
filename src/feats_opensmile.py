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
        self.featset = self.util.config_val('FEATS', 'set', 'eGeMAPSv02')
        if self.featset == 'eGeMAPSv02':
            self.feature_set = opensmile.FeatureSet.eGeMAPSv02
        elif self.featset == 'ComParE_2016':
            self.feature_set = opensmile.FeatureSet.ComParE_2016
        elif self.featset == 'GeMAPSv01a':
            self.feature_set = opensmile.FeatureSet.GeMAPSv01a
        elif self.featset == 'eGeMAPSv01a':
            self.feature_set = opensmile.FeatureSet.eGeMAPSv01a
        else:
            self.util.error(f'unknown feature set: {self.featset}')        

    def extract(self):
        store = self.util.get_path('store')
        storage = f'{store}{self.name}_{self.featset}.pkl'
        extract = self.util.config_val('DATA', 'needs_feature_extraction', False)
        is_multi_index = False
        if extract or not os.path.isfile(storage):
            self.util.debug('extracting openSmile features, this might take a while...')
            smile = opensmile.Smile(
            feature_set= self.feature_set,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=5,)
            if isinstance(self.data_df.index, pd.MultiIndex):
                is_multi_index = True
                self.df = smile.process_index(self.data_df.index)
            else:
                self.df = smile.process_files(self.data_df.index)

            self.df.to_pickle(storage)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else:
            self.util.debug('reusing extracted OS features.')
            self.df = pd.read_pickle(storage)
        # drop the multiindex if it's not in the data
        if not is_multi_index:
            self.df.index = self.df.index.droplevel(1)
            self.df.index = self.df.index.droplevel(1)


    def extract_sample(self, signal, sr):
        smile = opensmile.Smile(
                feature_set=self.feature_set,
                feature_level=opensmile.FeatureLevel.Functionals,)
        feats = smile.process_signal(signal, sr)
        return feats