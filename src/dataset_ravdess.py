# dataset.py
import audformat
import pandas as pd
import ast
import os
from random import sample
from util import Util
import glob_conf
from dataset import Dataset

class Ravdess(Dataset):
    """Class to represent the Berlin EmoDB"""
    name = 'ravdess' # The name    

    def __init__(self):
        """Constructor setting the name"""
        Dataset.__init__(self, self.name) 

    def load(self):
        Dataset.load(self) 
        df = self.df
        prev = df.shape[0]
        df = df[~df.index.str.contains('song')]
        now = df.shape[0]
        self.util.debug(f'removed {prev-now} songs from ravdess dataframe')
        self.df = df
