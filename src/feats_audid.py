# opensmileset.py
from featureset import Featureset
import pandas as pd
from util import Util 
import glob_conf
import audid


class AudIDset(Featureset):

    def extract(self):
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        try:
            extract = glob_conf.config['DATA']['needs_feature_extraction']
        except KeyError:
            extract = False
        if extract or not os.path.isfile(storage):
            print('extracting openSmile features, this might take a while...')
            feature_extractor = audid.Embedding(num_workers=4, verbose=False)
            self.df = smile.process_files(self.data_df.index)
            self.df.to_pickle(storage)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else:
            self.df = pd.read_pickle(storage)
        # drop the multiindex
        self.df.index = self.df.index.droplevel(1)
        self.df.index = self.df.index.droplevel(1)
