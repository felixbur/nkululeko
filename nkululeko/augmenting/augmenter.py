# augmenter.py
import os
import numpy as np
import pandas as pd
from audiomentations import *
from tqdm import tqdm
import audeer
import audiofile
from audformat.utils import map_file_path
from nkululeko.utils.util import Util


class Augmenter:
    """
    augmenting the train split
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("augmenter")
        # Define a standard transformation that randomly add augmentations to files
        # self.audioment = Compose(
        #     [
        #         AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        #         TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        #         PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        #         Shift(p=0.5),
        #     ]
        # )
        defaults = 'Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.05),Shift(p=0.5),BandPassFilter(min_center_freq=100.0, max_center_freq=6000),Limiter(min_threshold_db=-16.0,max_threshold_db=-6.0,threshold_mode="relative_to_signal_peak"),ClippingDistortion(),])'
        audiomentations = self.util.config_val("AUGMENT", "augmentations", defaults)
        self.audioment = eval(audiomentations)

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

        file_index = df_ret.index.to_series().map(lambda x: index_map[x[0]]).values
        # workaround because i just couldn't get this easier...
        arrays = [
            file_index,
            list(df_ret.index.get_level_values(1)),
            list(df_ret.index.get_level_values(2)),
        ]
        new_index = pd.MultiIndex.from_arrays(arrays, names=("file", "start", "end"))
        df_ret = df_ret.set_index(new_index)

        return df_ret
