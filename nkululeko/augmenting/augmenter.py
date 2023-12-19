# augmenter.py
import os

import audeer
import audiofile
import numpy as np
import pandas as pd
from audformat.utils import map_file_path
from audiomentations import (
    AddGaussianNoise,
    AddGaussianSNR,
    Compose,
    PitchShift,
    Shift,
    TimeStretch,
)
from nkululeko.utils.util import Util
from tqdm import tqdm


class Augmenter:
    """
    augmenting the train split
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("augmenter")
        # Define a standard transformation that randomly add augmentations to files
        self.audioment = Compose(
            [
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                Shift(p=0.5),
            ]
        )

    def changepath(self, fp, np):
        #        parent = os.path.dirname(fp).split('/')[-1]
        fullpath = os.path.dirname(fp)
        #       newpath = f'{np}{parent}'
        #       audeer.mkdir(newpath)
        return fp.replace(fullpath, np)

    def augment(self, sample_selection):
        """
        augment the training files and return a dataframe with new files index.
        """
        files = self.df.index.get_level_values(0).values
        store = self.util.get_path("store")
        filepath = f"{store}augmentations/"
        audeer.mkdir(filepath)
        self.util.debug(f"augmenting {sample_selection} samples to {filepath}")
        newpath = ""
        index_map = {}
        for i, f in enumerate(tqdm(files)):
            signal, sr = audiofile.read(f)
            filename = os.path.basename(f)
            parent = os.path.dirname(f).split("/")[-1]
            sig_aug = self.audioment(samples=signal, sample_rate=sr)
            newpath = f"{filepath}/{parent}/"
            audeer.mkdir(newpath)
            new_full_name = newpath + filename
            audiofile.write(new_full_name, signal=sig_aug, sampling_rate=sr)
            index_map[f] = new_full_name
        df_ret = self.df.copy()
        file_index = df_ret.index.levels[0].map(lambda x: index_map[x]).values
        df_ret = df_ret.set_index(df_ret.index.set_levels(file_index, level="file"))

        return df_ret
