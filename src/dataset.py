# dataset.py
import audformat
import pandas as pd
import ast

class Dataset:
    name = ''
    config = None
    df = None
    df_train = None
    df_test = None

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.load()
        self.prepare_labels()



    def load(self):
        pass

    def split_percent_speakers(self, test_percent):
        df = self.df
        s_num = df.speaker.nunique()
        test_num = int(s_num * (test_percent/100))
        train_spkrs = df.speaker.unique()[test_num:]
        test_spkrs = df.speaker.unique()[:test_num]
        self.df_train = df[df.speaker.isin(train_spkrs)]
        self.df_test = df[df.speaker.isin(test_spkrs)]

    def prepare_labels(self):
        mapping = ast.literal_eval(self.config['DATA'][f'{self.name}.mapping'])
        target = self.config['DATA']['target']
        labels = ast.literal_eval(self.config['DATA']['labels'])
        df = self.df
        df[target] = df[target].map(mapping)
        self.df = df[df[target].isin(labels)]
        print(f'for dataset {self.name} mapped {mapping}')
