# dataset_csv.py
import pandas as pd
import ast
import os
from random import sample
from dataset import Dataset
from util import Util
import glob_conf
from dataset import Dataset

class Dataset_CSV(Dataset):
    """ Class to represent datasets stored as a csv file"""

    def load(self):
        """Load the dataframe with files, speakers and task labels"""
        got_target, got_speaker, got_gender = False, False, False
        root = glob_conf.config['DATA'][self.name]
        self.util.debug(f'loading {self.name}')
        df = pd.read_csv(root, index_col='file')        
        self.df = df
        self.db = None
        got_target = True
        if 'gender' in df.columns:
            got_gender = True 
        if 'speaker' in df.columns:
            got_speaker = True
        self.util.debug(f'Loaded database {self.name} with {df.shape[0]} '\
            f'samples: got targets: {got_target}, got speakers: {got_speaker}, '\
            f'got sexes: {got_gender}')
