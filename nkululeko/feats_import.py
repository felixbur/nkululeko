# feats_import.py

from nkululeko.util import Util
from nkululeko.featureset import Featureset
import os
import pandas as pd
import audformat

class Importset(Featureset):
    """Class to import features that have been compiled elsewhere"""

    def __init__(self, name, data_df):
        super().__init__(name, data_df)

    def extract(self):
        """Import the features or load them from disk if present."""
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        extract = eval(self.util.config_val('FEATS', 'needs_feature_extraction', False))
        no_reuse = eval(self.util.config_val('FEATS', 'no_reuse', 'False'))
        feat_import_file = self.util.config_val('FEATS', 'import_file', False)
        if not os.path.isfile(feat_import_file):
            self.util.warn(f'no import file: {feat_import_file}')
        if extract or no_reuse or not os.path.isfile(storage):
            self.util.debug(f'importing features for {self.name}')
            # df = pd.read_csv(feat_import_file, sep=',', header=0, 
            #     index_col=['file', 'start', 'end'])
            df = audformat.utils.read_csv(feat_import_file)
            # scale features before use?
            # from sklearn.preprocessing import StandardScaler
            # scaler = StandardScaler()
            # scaled_features = scaler.fit_transform(df.values)
            # df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
            # use only the rows from the data index
            #df = self.data_df.join(df).drop(columns=self.data_df.columns)
            df = df.loc[self.data_df.index]
            #df = pd.concat([self.data_df, df], axis=1, join="inner").drop(columns=self.data_df.columns)
            # in any case, store to disk for later use
            df.to_pickle(storage) 
            # and assign to be the "official" feature set
            self.df = df           
        else:
            self.util.debug('reusing imported features.')
            self.df = pd.read_pickle(storage)
