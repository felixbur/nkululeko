from opensmile.core.define import FeatureSet
from dataset import Dataset
from opensmileset import Opensmileset
from runmanager import Runmanager

import pandas as pd

class Experiment:
    name = ''
    datasets = []
    df_train = None
    df_test = None 
    feats_train = None
    feats_test = None 
    config = None
    runmgr = None 

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.datasets = []

    def add_dataset(self, ds):
        self.datasets.append(ds)

    def fill_train_and_tests(self):
        self.df_train, self.df_test = pd.DataFrame(), pd.DataFrame()
        for d in self.datasets:
            d.split_percent_speakers(50)
            self.df_train = self.df_train.append(d.df_train)
            self.df_test = self.df_test.append(d.df_test)

    def extract_feats(self):
        df_train, df_test = self.df_train, self.df_test
        self.feats_train = Opensmileset(f'{self.name}_feats_train', self.config, df_train)
        self.feats_train.extract()
        self.feats_test = Opensmileset(f'{self.name}_feats_test', self.config, df_test)
        self.feats_test.extract()

    def init_runmanager(self):
        self.runmgr = Runmanager(self.config, self.df_train, self.df_test, self.feats_train, self.feats_test)

    def run(self):
        self.runmgr.do_runs()