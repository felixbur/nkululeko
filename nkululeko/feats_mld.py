# mld_fset.py
from nkululeko.featureset import Featureset
import sys
import os
import pandas as pd
import numpy as np
from nkululeko.util import Util 
import nkululeko.glob_conf as glob_conf

class MLD_set(Featureset):

    def __init__(self, name, data_df):
        self.name = name
        self.data_df = data_df
        self.util = Util()
        mld_path = self.util.config_val('FEATS', 'mld.model', None)
        sys.path.append(mld_path)

    def extract(self):
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        no_reuse = eval(self.util.config_val('FEATS', 'no_reuse', 'False'))
        if no_reuse:
            os.remove(storage)
        if not os.path.isfile(storage):
            self.util.debug('extracting midleveldescriptor features, this might take a while...')
        else:
            self.util.debug('reusing previously extracted midleveldescriptor features')
        import midlevel_descriptors as mld
        fex_mld = mld.MLD()
        self.df = fex_mld.extract_from_index(index=self.data_df, cache_path=storage)
        self.util.debug(f'MLD feats shape: {self.df.shape}')
        # shouldn't happen
        # replace NANa with column means values
        self.util.debug('MLD extractor: checking for NANs...')
        for i, col in enumerate(self.df.columns):
            if self.df[col].isnull().values.any():
                self.util.debug(f'{col} includes {self.df[col].isnull().sum()} nan, inserting mean values')
                self.df[col] = self.df[col].fillna(self.df[col].mean())

        try:
            # use only samples that have a minimum number of syllables
            min_syls = int(glob_conf.config['FEATS']['min_syls'])
            self.df = self.df[self.df['hld_nSyl']>=min_syls]
        except KeyError:
            pass
        if self.df.isna().to_numpy().any():
            self.util.error('feats 0: NANs exist')
        self.df = self.df.astype(float)
