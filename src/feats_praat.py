# feats_praat.py
from featureset import Featureset
import os
import pandas as pd
import glob_conf
import feinberg_praat
import ast

class Praatset(Featureset):
    """
    a feature extractor for the Praat software, based on 
    David R. Feinberg's Praat scripts for the parselmouth python interface.
    https://osf.io/6dwr3/

    """
    def __init__(self, name, data_df):
        super().__init__(name, data_df)

    def extract(self):
        """Extract the features based on the initialized dataset or re-open them when found on disk."""
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        extract = self.util.config_val('FEATS', 'needs_feature_extraction', False)
        start_fresh = self.util.config_val('DATA', 'no_reuse', False)
        if extract or start_fresh or not os.path.isfile(storage):
            self.util.debug('extracting Praat features, this might take a while...')
            self.df = feinberg_praat.compute_features(self.data_df.index)
            self.df = self.df.set_index(self.data_df.index)
            self.df.to_pickle(storage)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else:
            self.util.debug('reusing extracted Praat features.')
            self.df = pd.read_pickle(storage)
        self.util.debug(f'praat feature names: {self.df.columns}')
        self.df = self.df.astype(float)



    def extract_sample(self, signal, sr):
        self.util.error('feats_praat: not implemented yet')
        feats = None
        return feats