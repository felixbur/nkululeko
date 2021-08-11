# emodb.py

from dataset import Dataset
import audformat
import os

class Emodb(Dataset):
    """Class to represent the Berlin EmoDB"""
    name = 'emodb' # The name

    def __init__(self, config):
        """Constructor setting the name"""
        Dataset.__init__(self, self.name, config) 

    def load(self):
        """Load the dataframe with files, speakers and emotion labels"""
        root = self.config['DATA'][self.name]
        db = audformat.Database.load(root)
        db.map_files(lambda x: os.path.join(root, x))    
        df_emotion = db.tables['emotion'].df
        df = db.tables['files'].df
        # copy the emotion label from the the emotion dataframe to the files dataframe
        df['emotion'] = df_emotion['emotion']
        self.df = df
        self.db = db


    def split(self):
        """One way to split train and eval sets: Specify percentage of evaluation speakers"""
        test_percent = int(self.config['DATA'][self.name+'.testsplit'])
        df = self.df
        s_num = df.speaker.nunique()
        test_num = int(s_num * (test_percent/100))
        train_spkrs = df.speaker.unique()[test_num:]
        test_spkrs = df.speaker.unique()[:test_num]
        self.df_train = df[df.speaker.isin(train_spkrs)]
        self.df_test = df[df.speaker.isin(test_spkrs)]

