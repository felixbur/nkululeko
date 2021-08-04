# emodb.py

from dataset import Dataset
import audformat
import os

class Emodb(Dataset):
    name = 'emodb'
    def __init__(self, config):
        Dataset.__init__(self, self.name, config) 

    def load(self):
        root = self.config['data'][self.name]
        db = audformat.Database.load(root)
        db.map_files(lambda x: os.path.join(root, x))    
        df_emotion = db.tables['emotion'].df
        df = db.tables['files'].df
        # copy the emotion label from the the emotion dataframe to the files dataframe
        df['emotion'] = df_emotion['emotion']
        self.df = df