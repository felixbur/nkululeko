# feats_oxbow.py

from util import Util
from featureset import Featureset
import os
import pandas as pd
import opensmile

class Openxbow(Featureset):
    """Class to extract openXBOW processed opensmile features (https://github.com/openXBOW)"""



    def extract(self):
        """Extract the features or load them from disk if present."""
        self.featset = self.util.config_val('FEATS', 'set', 'eGeMAPSv02')
        self.feature_set = eval(f'opensmile.FeatureSet.{self.featset}')
        store = self.util.get_path('store')
        storage = f'{store}{self.name}_{self.featset}.pkl'
        extract = self.util.config_val('DATA', 'needs_feature_extraction', False)
        is_multi_index = False
        if extract or not os.path.isfile(storage):
            # extract smile features first
            self.util.debug('extracting openSmile features, this might take a while...')
            smile = opensmile.Smile(
                feature_set= self.feature_set,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                num_workers=5,)
            if isinstance(self.data_df.index, pd.MultiIndex):
                is_multi_index = True
                smile_df = smile.process_index(self.data_df.index)
            else:
                smile_df = smile.process_files(self.data_df.index)
            smile_df.index = smile_df.index.droplevel(1)
            smile_df.index = smile_df.index.droplevel(1)
            print(smile_df.head(10))
            lld_name, xbow_name = 'llds.csv', 'xbow.csv'
            smile_df.to_csv(lld_name, sep=';')
            xbow_path = self.util.config_val('FEATS', 'xbow', './openXBOW/')
            os.system(f'java -jar {xbow_path}openXBOW.jar -i {lld_name} -o {xbow_name}')
        else:
            self.util.debug('reusing extracted OS features.')
            self.df = pd.read_pickle(storage)
