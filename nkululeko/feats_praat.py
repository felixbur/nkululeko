# feats_praat.py
from nkululeko.featureset import Featureset
import os
import pandas as pd
import nkululeko.glob_conf as glob_conf
from nkululeko import feinberg_praat

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
        store_format = self.util.config_val('FEATS', 'store_format', 'pkl')
        storage = f'{store}{self.name}.{store_format}'
        extract = self.util.config_val('FEATS', 'needs_feature_extraction', False)
        no_reuse = eval(self.util.config_val('FEATS', 'no_reuse', 'False'))
        if extract or no_reuse or not os.path.isfile(storage):
            self.util.debug('extracting Praat features, this might take a while...')
            self.df = feinberg_praat.compute_features(self.data_df.index)
            self.df = self.df.set_index(self.data_df.index)
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else:
            self.util.debug('reusing extracted Praat features.')
            self.df = self.util.get_store(storage, store_format)
        self.util.debug(f'praat feature names: {self.df.columns}')
        self.df = self.df.astype(float)



    def extract_sample(self, signal, sr):
        self.util.error('feats_praat: extracting single samples not implemented yet')
        feats = None
        return feats