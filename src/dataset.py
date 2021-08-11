# dataset.py
import audformat
import pandas as pd
import ast
import os
from sklearn.preprocessing import LabelEncoder

class Dataset:
    """ Class to represent datasets"""
    name = '' # An identifier for the dataset
    config = None # The configuration 
    db = None # The database object
    df = None # The whole dataframe
    df_train = None # The training split
    df_test = None # The evaluation split

    def __init__(self, config, name):
        """Constructor setting up name and configuration"""
        self.name = name
        self.config = config
        self.target = config['DATA']['target']

    def load(self):
        """Load the dataframe with files, speakers and emotion labels"""
        root = self.config['DATA'][self.name]
        db = audformat.Database.load(root)
        # map the audio file paths 
        db.map_files(lambda x: os.path.join(root, x))
        # the dataframe with the targets, e.g. emotion or age    
        df_target = db.tables[self.target].df
        # the dataframe with all other information 
        df = db.tables['files'].df
        # copy the targets to the main dataframe
        df[self.target] = df_target[self.target]
        self.df = df
        self.db = db

    def split(self):
        """Split the datbase into train and devel set"""
        testdf = self.db.tables[f'files.dev'].df
        traindf = self.db.tables[f'files.train'].df
        self.df_test = self.df.loc[self.df.index.intersection(testdf.index)]
        self.df_train = self.df.loc[self.df.index.intersection(traindf.index)]

    def prepare_labels(self):
        """Rename the labels and remove the ones that are not needed."""
        mapping = ast.literal_eval(self.config['DATA'][f'{self.name}.mapping'])
        target = self.config['DATA']['target']
        labels = ast.literal_eval(self.config['DATA']['labels'])
        df = self.df
        df[target] = df[target].map(mapping)
        self.df = df[df[target].isin(labels)]
        print(f'for dataset {self.name} mapped {mapping}')