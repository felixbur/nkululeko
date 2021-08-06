# dataset.py
import audformat
import pandas as pd
import ast

class Dataset:
    """ Class to represent datasets"""

    name = '' # An identifier for the dataset
    config = None # The configuration 
    df = None # The whole dataframe
    df_train = None # The training split
    df_test = None # The evaluation split

    def __init__(self, name, config):
        """Constructor setting up name and configuration"""
        self.name = name
        self.config = config

    def load(self):
        """To be implemented by inheriting classes."""
        pass

    def split_percent_speakers(self, test_percent):
        """One way to split train and eval sets: Specify percentage of evaluation speakers"""
        df = self.df
        s_num = df.speaker.nunique()
        test_num = int(s_num * (test_percent/100))
        train_spkrs = df.speaker.unique()[test_num:]
        test_spkrs = df.speaker.unique()[:test_num]
        self.df_train = df[df.speaker.isin(train_spkrs)]
        self.df_test = df[df.speaker.isin(test_spkrs)]

    def prepare_labels(self):
        """Rename the labels and remove the ones that are not needed."""
        mapping = ast.literal_eval(self.config['DATA'][f'{self.name}.mapping'])
        target = self.config['DATA']['target']
        labels = ast.literal_eval(self.config['DATA']['labels'])
        df = self.df
        df[target] = df[target].map(mapping)
        self.df = df[df[target].isin(labels)]
        print(f'for dataset {self.name} mapped {mapping}')