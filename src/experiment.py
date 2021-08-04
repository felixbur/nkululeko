from dataset import Dataset
import pandas as pd

class Experiment:
    name = ''
    datasets = []
    df_train = None
    df_test = None 

    def __init__(self, name):
        self.name = name
        self.datasets = []

    def add_dataset(self, ds):
        self.datasets.append(ds)

    def fill_train_and_tests(self):
        self.df_train, self.df_test = pd.DataFrame(), pd.DataFrame()
        for d in self.datasets:
            d.split_percent_speakers(50)
            self.df_train = self.df_train.append(d.df_train)
            self.df_test = self.df_test.append(d.df_test)
