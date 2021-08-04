from opensmile.core.define import FeatureSet
from dataset import Dataset
from emodb import Emodb
from opensmileset import Opensmileset
from runmanager import Runmanager
import ast
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
    labels = None # set of string values for the categories
    values = None # set of numerical values encoding the classes 

    def __init__(self, name, config):
        self.name = name
        self.config = config

    def load_datasets(self):
        ds = ast.literal_eval(self.config['DATA']['databases'])
        for d in ds:
            if d == 'emodb':
                data = Emodb(self.config)
            data.load()
            data.prepare_labels()
            print(f'check experiment: {data.df.emotion.unique()}')
            self.datasets.append(data)
        print(f'check 2 {self.datasets[0].df.emotion.unique()}')

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