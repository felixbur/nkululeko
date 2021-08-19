# dataset.py
import audformat
import pandas as pd
import ast
import os
from sklearn.preprocessing import LabelEncoder
from random import sample
from util import Util
import glob_conf

class Dataset:
    """ Class to represent datasets"""
    name = '' # An identifier for the dataset
    config = None # The configuration 
    db = None # The database object
    df = None # The whole dataframe
    df_train = None # The training split
    df_test = None # The evaluation split

    def __init__(self, name):
        """Constructor setting up name and configuration"""
        self.name = name
        self.target = glob_conf.config['DATA']['target']
        self.util = Util()

    def load(self):
        """Load the dataframe with files, speakers and emotion labels"""
        root = glob_conf.config['DATA'][self.name]
        db = audformat.Database.load(root)
        # map the audio file paths 
        db.map_files(lambda x: os.path.join(root, x))
        # the dataframe with all other information 
        df = db.tables['files'].df
        try :
           # There might be a separate table with the targets, e.g. emotion or age    
            df_target = db.tables[self.target].df
            df[self.target] = df_target[self.target]
        except KeyError:
            pass
        try: 
            # for experiments that do separate sex models
            s = glob_conf.config['DATA']['sex']
            df = df[df.gender==s]
        except KeyError:
            pass 
        self.df = df
        self.db = db

    def split(self):
        """Split the datbase into train and development set"""
        store = self.util.get_path('store')
        storage_test = f'{store}{self.name}_testdf.pkl'
        storage_train = f'{store}{self.name}_traindf.pkl'
        try:
            split_strategy = glob_conf.config['DATA'][self.name+'.split_strategy']
        except KeyError:
            split_strategy = 'database'
        # 'database' (default), 'speaker_split', 'specified', 'reuse'
        if split_strategy == 'database':
            #  use the splits from the database
            testdf = self.db.tables[self.target+'.test'].df
            traindf = self.db.tables[self.target+'.train'].df
            # use only the train and test samples that were not perhaps filtered out by an earlier processing step
            self.df_test = self.df.loc[self.df.index.intersection(testdf.index)]
            self.df_train = self.df.loc[self.df.index.intersection(traindf.index)]
        elif split_strategy == 'specified':
            test_table = glob_conf.config['DATA'][self.name+'.test_table']
            train_table = glob_conf.config['DATA'][self.name+'.train_table']
            testdf = self.db.tables[test_table].df
            traindf = self.db.tables[train_table].df
            # use only the train and test samples that were not perhaps filtered out by an earlier processing step
            self.df_test = self.df.loc[self.df.index.intersection(testdf.index)]
            self.df_train = self.df.loc[self.df.index.intersection(traindf.index)]
        elif split_strategy == 'speaker_split':
            self.split_speakers()
        elif split_strategy == 'reuse':
            self.df_test = pd.read_pickle(storage_test)
            self.df_train = pd.read_pickle(storage_train)
        # remember the splits for future use
        self.df_test.to_pickle(storage_test)
        self.df_train.to_pickle(storage_train)
        

    def split_speakers(self):
        """One way to split train and eval sets: Specify percentage of evaluation speakers"""
        test_percent = int(self.util.config_val('DATA', self.name+'.testsplit', 50))
        df = self.df
        s_num = df.speaker.nunique()
        test_num = int(s_num * (test_percent/100))        
        test_spkrs =  sample(list(df.speaker.unique()), test_num)
        self.df_test = df[df.speaker.isin(test_spkrs)]
        self.df_train = df[~df.index.isin(self.df_test.index)]
        # because this generates new train/test sample quantaties, the feature extraction has to be done again
        glob_conf.config['DATA']['needs_feature_extraction'] = 'true'

    def prepare_labels(self):
        """Rename the labels and remove the ones that are not needed."""
        try :
            mapping = ast.literal_eval(glob_conf.config['DATA'][f'{self.name}.mapping'])
            target = glob_conf.config['DATA']['target']
            labels = ast.literal_eval(glob_conf.config['DATA']['labels'])
            df = self.df
            df[target] = df[target].map(mapping)
            self.df = df[df[target].isin(labels)]
            print(f'for dataset {self.name} mapped {mapping}')
        except KeyError:
            pass
