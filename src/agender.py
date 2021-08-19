# agender.py

from dataset import Dataset
import audformat
import os
import glob_conf

class Agender(Dataset):
    """Class to represent the agender age and gender dataset"""
    name = 'agender' # The name

    def __init__(self, config):
        """Constructor setting the name"""
        Dataset.__init__(self, self.name, config) 

    def load(self):
        """Load the dataframe with files, speakers and emotion labels"""
        root = self.config['DATA'][self.name]
        db = audformat.Database.load(root)
        db.map_files(lambda x: os.path.join(root, x))    
        self.db = db

    def split(self):
        """Split the datbase into train and devel set"""
        self.df_test = self.db.tables['age.devel'].df
        self.df_train = self.db.tables['age.train'].df