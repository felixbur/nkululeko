# feats_praat.py
from featureset import Featureset
import opensmile
import os
import pandas as pd
from util import Util 
import glob_conf

class Praatset(Featureset):
    """
    a feature extractor for the Praat software, based on 
    David R. Feinberg's Praat scripts for the parselmouth python interface.
    https://osf.io/6dwr3/

    """
    def __init__(self, name, data_df)
        super().__init__(name, data_df)

    def extract(self):
        """Extract the features based on the initialized dataset or re-open them when found on disk."""
        store = self.util.get_path('store')
        storage = f'{store}{self.name}_{self.featset}.pkl'
        extract = self.util.config_val('FEATS', 'needs_feature_extraction', False)
        if extract or not os.path.isfile(storage):
            self.util.debug('extracting Praat features, this might take a while...')
            if isinstance(self.data_df.index, pd.MultiIndex):

            else:

            self.df.to_pickle(storage)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else:
            self.util.debug('reusing extracted Praat features.')
            self.df = pd.read_pickle(storage)


    def extract_sample(self, signal, sr):
        smile = opensmile.Smile(
                feature_set=self.feature_set,
                feature_level=opensmile.FeatureLevel.Functionals,)
        feats = smile.process_signal(signal, sr)
        return feats