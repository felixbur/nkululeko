"""augmenter_auglib.py

augmentations implemented with auglib
-> https://audeering.github.io/auglib/

"""

import os

import audeer
import audiofile
import auglib
import audb
import ast
import pandas as pd
from tqdm import tqdm
from nkululeko.utils.util import Util
from nkululeko.utils.dataframe import remap_augmented_index
from nkululeko.utils.files import mirror_relpath
from nkululeko.constants import SAMPLING_RATE


class AugmenterAuglib:
    """
    augmenting the train split
    """

    def __init__(self, df):
        self.df = df
        self.util = Util("augmenter_auglib")
        transforms = self.util.config_val(
            "AUGMENT",
            "transformations",
            '["room", "music", "noise", "babble", "crop", "cough"]',
        )
        self.util.debug(f"applying transformations: {transforms}")
        transforms = ast.literal_eval(transforms)
        transformations = []
        bypass_prob = float(self.util.config_val("AUGMENT", "bypass_prob", 0.3))
        # Each external asset is only downloaded if its transform is
        # actually selected -- e.g. requesting just ["noise", "babble",
        # "room"] must not also pull the "music" table nobody asked for.
        if "cough" in transforms:
            cough_files = audb.load_media(
                "cough-speech-sneeze",
                [
                    "coughing/sqei2xjfnpk_262.51-263.43.wav",
                    "coughing/3id3zrrzbvm_139.88-140.83.wav",
                    "coughing/kopzxumj430_40.94-41.8.wav",
                    "coughing/kopzxumj430_37.98-39.18.wav",
                    "coughing/ta_ihseeuyk_118.7-119.84.wav",
                    "coughing/_j7jejkncl4_0.84-1.39.wav",
                    "coughing/9kigihccwvq_84.26-85.26.wav",
                    "coughing/4wlf2ct0ecm_32.52-33.6.wav",
                    "coughing/_0rh6xgxhrq_53.41-55.87.wav",
                    "coughing/2mw_s5jnqxu_75.49-76.59.wav",
                    "coughing/dizkwd7jj_q_71.98-72.93.wav",
                    "coughing/ipo39x2bv9c_12.7253-13.7358.wav",
                    "coughing/vwe0wljpgyu_111.11-113.12.wav",
                ],
                version="2.0.1",
                sampling_rate=SAMPLING_RATE,
            )
            transformations.append(
                auglib.transform.Append(
                    auglib.observe.List(cough_files, draw=True),
                    bypass_prob=bypass_prob,
                )
            )
        if "room" in transforms:
            db_air = audb.load(
                "air",
                version="1.4.2",
                tables="rir",
                channels=[0],
                sampling_rate=16000,
                verbose=False,
            )
            transformations.append(
                auglib.transform.FFTConvolve(
                    auglib.observe.List(db_air.files, draw=True),
                    keep_tail=False,
                    preserve_level=True,
                    bypass_prob=bypass_prob,
                )
            )
        if "music" in transforms:
            db_musan_music = audb.load(
                "musan",
                tables="music",
                media="music/fma/music-fma-0097.wav",
                version="1.0.0",
                verbose=False,
            )
            transformations.append(
                auglib.transform.Mix(
                    auglib.observe.List(db_musan_music.files, draw=True),
                    gain_aux_db=auglib.observe.IntUni(-15, -10),
                    read_pos_aux=auglib.observe.FloatUni(0, 1),
                    unit="relative",
                    snr_db=10,
                    loop_aux=True,
                    bypass_prob=bypass_prob,
                )
            )
        if "babble" in transforms:
            db_musan_speech = audb.load(
                "musan",
                tables="speech",
                media=".*speech-librivox-000\\d",
                version="1.0.0",
                verbose=False,
            )
            transformations.append(
                auglib.transform.BabbleNoise(
                    list(db_musan_speech.files),
                    num_speakers=auglib.observe.IntUni(3, 7),
                    snr_db=auglib.observe.IntUni(13, 20),
                    bypass_prob=bypass_prob,
                )
            )
        if "noise" in transforms:
            transformations.append(
                auglib.transform.PinkNoise(
                    snr_db=10,
                    bypass_prob=bypass_prob,
                )
            )
        if "crop" in transforms:
            crop_dur = float(self.util.config_val("AUGMENT", "crop_dur", 1.0))
            transformations.append(
                auglib.transform.Trim(
                    start_pos=auglib.Time(
                        auglib.observe.FloatUni(0, 1), unit="relative"
                    ),
                    duration=crop_dur,
                    fill="loop",
                    unit="seconds",
                    bypass_prob=bypass_prob,
                )
            )
        transformations.append(auglib.transform.NormalizeByPeak())
        transform = auglib.transform.Compose(
            transformations,
        )
        self.augmenter = auglib.Augment(transform)

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
        filepath = f"{store}auglib/"
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
            sig_aug = self.augmenter(signal, sr)
            audiofile.write(new_full_name, signal=sig_aug, sampling_rate=sr)
            index_map[f] = new_full_name

        return remap_augmented_index(self.df, index_map)
