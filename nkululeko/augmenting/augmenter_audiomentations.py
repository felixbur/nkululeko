# augmenter.py
import os

import audeer
import audiofile
import pandas as pd
from audiomentations import *  # noqa: F403
from tqdm import tqdm

from nkululeko.utils.dataframe import remap_augmented_index
from nkululeko.utils.files import mirror_relpath
from nkululeko.utils.util import Util


class AugmenterAudiomentations:
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
        # dedupe: a segmented index can list the same file for multiple
        # (start, end) rows, and augmentation is a per-file operation --
        # without this, each segment would trigger its own (randomized)
        # augmentation run, wasting work and leaving whichever run for that
        # file happened last mapped to every one of its segments.
        files = pd.unique(self.df.index.get_level_values(0).values)
        store = self.util.get_path("store")
        filepath = f"{store}augmentations/"
        audeer.mkdir(filepath)
        self.util.debug(f"augmenting {sample_selection} samples to {filepath}")
        index_map = {}
        for f in tqdm(files):
            signal, sr = audiofile.read(f)
            # Keyed by the full source path (not just the immediate parent
            # directory name) so two datasets that happen to share a
            # subfolder name and a filename don't collide and silently
            # overwrite each other's augmented file.
            new_full_name = os.path.join(filepath, mirror_relpath(f))
            audeer.mkdir(os.path.dirname(new_full_name))
            sig_aug = self.audioment(samples=signal, sample_rate=sr)
            audiofile.write(new_full_name, signal=sig_aug, sampling_rate=sr)
            index_map[f] = new_full_name

        return remap_augmented_index(self.df, index_map)
