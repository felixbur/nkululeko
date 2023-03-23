# augmenter.py
import pandas as pd
from nkululeko.util import Util 
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import audiofile
import os
from audformat.utils import map_file_path
import audeer

class Augmenter:
    """
        augmenting the train split
    """
    def __init__(self, train_df):
        self.train_df = train_df
        self.util = Util()
        # Define a standard transformation that randomly add augmentations to files
        self.audioment = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])

    def changepath(self, fp, np):
#        parent = os.path.dirname(fp).split('/')[-1]
        fullpath = os.path.dirname(fp)
 #       newpath = f'{np}{parent}'
 #       audeer.mkdir(newpath)
        return fp.replace(fullpath, np)

    def augment(self):
        """
        augment the training files and return a dataframe with new files index.
        """
        files = self.train_df.index.get_level_values(0).values
        store = self.util.get_path('store')
        filepath = f'{store}augmentations/'
        audeer.mkdir(filepath)
        self.util.debug(f'augmenting the training set to {filepath}')
        newpath = ''
        for i, f in enumerate(files):
            signal, sr = audiofile.read(f)
            filename = os.path.basename(f)
            parent = os.path.dirname(f).split('/')[-1]
            sig_aug = self.audioment(samples = signal, sample_rate = sr)
            newpath = f'{filepath}/{parent}/'
            audeer.mkdir(newpath)
            audiofile.write(f'{newpath}{filename}', signal=sig_aug, sampling_rate=sr)
            if i%10==0:
                print(f'augmented {i} of {len(files)}')
        df_ret = self.train_df.copy()
        df_ret = df_ret.set_index(map_file_path(df_ret.index, lambda x: self.changepath(x, newpath)))
        aug_db_filename = self.util.config_val('DATA', 'augment', 'augment.csv')
        target = self.util.config_val('DATA', 'target', 'emotion')
        df_ret[target] = df_ret['class_label']
        df_ret = df_ret.drop(columns=['class_label'])
        df_ret.to_csv(aug_db_filename)
        return df_ret