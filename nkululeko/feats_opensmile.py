# opensmileset.py
from nkululeko.featureset import Featureset
import opensmile
import os
import pandas as pd
import nkululeko.glob_conf as glob_conf


class Opensmileset(Featureset):

    def __init__(self, name, data_df):
        super().__init__(name, data_df)
        self.featset = self.util.config_val('FEATS', 'set', 'eGeMAPSv02')
        try:
            self.feature_set = eval(f'opensmile.FeatureSet.{self.featset}')
            #'eGeMAPSv02, ComParE_2016, GeMAPSv01a, eGeMAPSv01a':
        except AttributeError:        
            self.util.error(f'something is wrong with feature set: {self.featset}')        
        self.featlevel = self.util.config_val('FEATS', 'level', 'functionals')
        try:
            self.featlevel = self.featlevel.replace('lld', 'LowLevelDescriptors')
            self.featlevel = self.featlevel.replace('functionals', 'Functionals')
            self.feature_level = eval(f'opensmile.FeatureLevel.{self.featlevel}')
        except AttributeError:        
            self.util.error(f'something is wrong with feature level: {self.featlevel}')        


    def extract(self):
        """Extract the features based on the initialized dataset or re-open them when found on disk."""
        store = self.util.get_path('store')
        storage = f'{store}{self.name}_{self.featset}.pkl'
        extract = eval(self.util.config_val('FEATS', 'needs_feature_extraction', 'False'))
        no_reuse = eval(self.util.config_val('FEATS', 'no_reuse', 'False'))
        if extract or not os.path.isfile(storage) or no_reuse:
            self.util.debug('extracting openSmile features, this might take a while...')
            smile = opensmile.Smile(
            feature_set= self.feature_set,
            feature_level=self.feature_level,
            num_workers=5,)
            if isinstance(self.data_df.index, pd.MultiIndex):
                self.df = smile.process_index(self.data_df.index)
                self.df = self.df.set_index(self.data_df.index)
            else:
                self.df = smile.process_files(self.data_df.index)
                self.df.index = self.df.index.droplevel(1)
                self.df.index = self.df.index.droplevel(1)
            self.df.to_pickle(storage)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'False'
            except KeyError:
                pass
        else:
            self.util.debug('reusing extracted OS features.')
            self.df = pd.read_pickle(storage)


    def extract_sample(self, signal, sr):
        smile = opensmile.Smile(
                feature_set=self.feature_set,
                feature_level=opensmile.FeatureLevel.Functionals,)
        feats = smile.process_signal(signal, sr)
        return feats.to_numpy()