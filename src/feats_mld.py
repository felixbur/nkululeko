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
            self.util.debug('extracting midleveldescriptor features, this might take a while...')
        else:
            self.util.debug('reusing previously extracted midleveldescriptor features')
        fex_mld = mld.MLD()
        self.df = fex_mld.extract_from_index(index=self.data_df, cache_path=storage)
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
        try: 
            # use only some features
            selected_features = ast.literal_eval(glob_conf.config['FEATS']['features'])
            self.df = self.df[selected_features]
        except KeyError:
            pass
        self.util.debug(f'MLD feats shape: {self.df.shape}')
        # add opensmile features
        with_os = self.util.config_val('FEATS', 'with_os', False)
        if with_os:
            df_os = self.extract_os()
            # df_os =  df_os.loc[ df_os.index.intersection(self.df.index)]
            self.df = pd.concat([self.df, df_os], axis=1)
            self.util.debug(f'new feats shape after adding OS featues: {self.df.shape}')
        # shouldn't happen
        if self.df.isna().to_numpy().any():
            self.util.error('feats 0: NANs exist')
        self.df = self.df.astype(float)

    def extract_os(self):
        store = self.util.get_path('store')
        name = self.name.replace('mld', 'os')
        storage = f'{store}{name}.pkl'
        extract = self.util.config_val('DATA', 'needs_feature_extraction', False)
        if extract or not os.path.isfile(storage):
            self.util.debug('extracting openSmile features, this might take a while...')
            smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.GeMAPSv01b,
            feature_level=opensmile.FeatureLevel.Functionals,)
            df = smile.process_files(self.data_df.index)
            df.to_pickle(storage)
            self.util.set_config_val('DATA', 'needs_feature_extraction', '0')
        else:
            self.util.debug('OS: reusing already extracted features.')
            df = pd.read_pickle(storage)
        # drop the multiindex  if still there
        try:
            df.index = df.index.droplevel(1)
            df.index = df.index.droplevel(1)
        except IndexError:
            pass
        # replace NANa with column means values
        self.util.debug('OS extractor: checking for NANs...')
        for i, col in enumerate(df.columns):
            if np.isnan(df[col]).any():
                self.util.debug(f'{col} includes {df[col].isna().sum()} nan, inserting mean values')
                df[col] = df[col].fillna(df[col].mean())
        return df