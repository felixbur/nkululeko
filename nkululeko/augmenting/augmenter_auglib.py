""" augmenter_auglib.py

augmentations implemented with auglib
-> https://audeering.github.io/auglib/

"""
import os

import audeer
import audiofile
import pandas as pd
import auglib 
import audb
import ast
from tqdm import tqdm

from nkululeko.utils.util import Util


class AugmenterAuglib:
    """
    augmenting the train split
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("augmenter_auglib")
        self.util.debug("loading databases for augmentation ...")
        db_air = audb.load(
            "air",
            version="1.4.2",
            tables="rir",
            channels=[0],
            sampling_rate=16000,
            verbose=False,
        )
        db_musan = audb.load(
            "musan",
            tables="music",
            media="music/fma/music-fma-0097.wav",
            version="1.0.0",
            verbose=False,
        )
        db_babble = audb.load(
            "musan",
            tables="speech",
            media=".*speech-librivox-000\d",
            version="1.0.0",
            verbose=False,
        )
        transforms = self.util.config_val("AUGMENT", "transformations", '["room", "music", "noise", "babble", "crop", "cough"]')
        self.util.debug(f"applying transformations: {transforms}")
        transforms = ast.literal_eval(transforms)
        transformations = []
        if "cough" in transforms:
            files = audb.load_media(
                "cough-speech-sneeze",
                "coughing/kopzxumj430_40.94-41.8.wav",
                version="2.0.1",
                sampling_rate=16000,
                )
            cough, _ = audiofile.read(files[0])
            transformations.append(auglib.transform.Append(cough))
        if "room" in transforms:
            transformations.append(auglib.transform.FFTConvolve(
                        auglib.observe.List(db_air.files, draw=True),
                        keep_tail=False,
                    ))
        if "music" in transforms:
            transformations.append(auglib.transform.Mix(
                    auglib.observe.List(db_musan.files, draw=True),
                    gain_aux_db=auglib.observe.IntUni(-15, -10),
                    read_pos_aux=auglib.observe.FloatUni(0, 1),
                    unit="relative",
                    snr_db=10,
                    loop_aux=True,
                )) 
        if "babble" in transforms:
            transformations.append(auglib.transform.BabbleNoise(
                list(db_babble.files),
                num_speakers=auglib.observe.IntUni(3, 7),
                snr_db=auglib.observe.IntUni(13, 20),)
                )
        if "noise" in transforms:
            transformations.append(auglib.transform.PinkNoise(snr_db=10))
        if "crop" in transforms:
            crop_dur = float(self.util.config_val("AUGMENT", "crop_dur", 1.0))
            transformations.append(auglib.transform.Trim(
                start_pos=auglib.Time(auglib.observe.FloatUni(0, 1), unit="relative"),
                duration=crop_dur,
                fill="loop",
                unit="seconds",
            ))
        bypass_prob = float(self.util.config_val("AUGMENT", "bypass_prob", 0.3))

        transformations.append(auglib.transform.NormalizeByPeak())
        transform = auglib.transform.Select(
            transformations,
            bypass_prob=bypass_prob,
        )
        self.augmenter = auglib.Augment(transform)

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
        filepath = f"{store}auglib/"
        audeer.mkdir(filepath)
        self.util.debug(f"augmenting {sample_selection} samples to {filepath}")
        newpath = ""
        index_map = {}
        for i, f in enumerate(tqdm(files)):
            signal, sr = audiofile.read(f)
            filename = os.path.basename(f)
            parent = os.path.dirname(f).split("/")[-1]
            sig_aug = self.augmenter(signal, sr)
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
