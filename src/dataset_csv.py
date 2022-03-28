# dataset_csv.py
import pandas as pd
import ast
import os
from random import sample
from dataset import Dataset
from util import Util
import glob_conf
from dataset import Dataset
import os.path


class Dataset_CSV(Dataset):
    """ Class to represent datasets stored as a csv file"""

    def load(self):
        """Load the dataframe with files, speakers and task labels"""
        self.got_target, self.got_speaker, self.got_gender = False, False, False
        root = self.util.config_val_data(self.name, '', '')
        self.util.debug(f'loading {self.name}')
        df = pd.read_csv(root, index_col='file')       
        # add the root folder to the relative paths of the files 
        df = df.set_index(df.index.to_series().apply(lambda x: os.path.dirname(root)+'/'+x))
        self.df = df
        self.db = None
        self.got_target = True
        self.is_labeled = self.got_target
        self.start_fresh = self.util.config_val('DATA', 'no_reuse', False)
        if 'gender' in df.columns:
            self.got_gender = True 
        if 'speaker' in df.columns:
            self.got_speaker = True
            ns = df['speaker'].nunique()
            print(f'num of speakers: {ns}')
        self.util.debug(f'Loaded database {self.name} with {df.shape[0]} '\
            f'samples: got targets: {self.got_target}, got speakers: {self.got_speaker}, '\
            f'got sexes: {self.got_gender}')
