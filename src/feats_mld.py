# mld_fset.py
from featureset import Featureset
import sys
sys.path.append("/home/felix/data/research/mld/src")
import os
import ast
import pandas as pd
import numpy as np
from util import Util 
import midlevel_descriptors as mld
import opensmile
import glob_conf

class MLD_set(Featureset):

    def extract(self):
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'
        if not os.path.isfile(storage):
            print('extracting midleveldescriptor features, this might take a while...')
        else:
            pass
        fex_mld = mld.MLD()
        self.df = fex_mld.extract_from_index(index=self.data_df, cache_path=storage)
        # replace NANa with column means values
        for i, col in enumerate(self.df.columns):
            if np.isnan(self.df[col]).any():
                print('{} includes {} nan, inserting mean values'.format(col, self.df[col].isna().sum()))
                self.df[col] = self.df[col].fillna(self.df[col].mean())

        try:
            # use only samples that have a minimum number of syllables
            min_syls = int(glob_conf.config['FEATS']['min_syls'])
            self.df = self.df[self.df['hld_nSyl']>=min_syls]
        except KeyError:
            pass
        try:
            # add opensmile features
            with_os = bool(glob_conf.config['FEATS']['with_os'])
            if with_os:
                df_os = self.extract_os()
                df_os =  df_os.loc[ df_os.index.intersection(self.df.index)]
                self.df = pd.concat([self.df, df_os], axis=1)
        except KeyError:
            pass
        try: 
            # use only some features
            selected_features = ast.literal_eval(glob_conf.config['FEATS']['features'])
            self.df = self.df[selected_features]
        except KeyError:
            pass

    def extract_os(self):
        store = self.util.get_path('store')
        storage = f'{store}{self.name}.pkl'.replace('mld', 'os')
        if not os.path.isfile(storage):
            print('extracting openSmile features, this might take a while...')
            smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,)
            df = smile.process_files(self.data_df.index)
            df.to_pickle(storage)
        else:
            df = pd.read_pickle(storage)
        # drop the multiindex  if still there
        try:
            df.index = self.df.index.droplevel(1)
            df.index = self.df.index.droplevel(1)
        except IndexError:
            pass
        return df