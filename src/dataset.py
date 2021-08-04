# dataset.py
import audformat
import pandas

class Dataset:
    name = ''
    config = None
    df = None
    df_train = None
    df_test = None

    def __init__(self, name, config):
        self.name = name
        self.config = config



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