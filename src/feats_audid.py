# opensmileset.py
from featureset import Featureset
import pandas as pd
from util import Util 
import glob_conf
import audid
import os

class AudIDset(Featureset):

    def extract(self):
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        try:
            extract = glob_conf.config['DATA']['needs_feature_extraction']
        except KeyError:
            extract = False
        if extract or not os.path.isfile(storage):
            print('extracting audid embeddings, this might take a while...')
            feature_extractor = audid.Embedding(num_workers=4, verbose=False)
            embeddings = feature_extractor.process_files(self.data_df.index, ends=5).to_numpy()
            self.df = pd.DataFrame(embeddings, index=self.data_df.index)
            self.df.to_pickle(storage)
            try:
                glob_conf.config['DATA']['needs_feature_extraction'] = 'false'
            except KeyError:
                pass
        else:
            self.df = pd.read_pickle(storage)