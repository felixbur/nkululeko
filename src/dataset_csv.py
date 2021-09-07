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
        root = glob_conf.config['DATA'][self.name]
        df = pd.read_csv(root, index_col='file')        
        self.df = df
        self.db = None
