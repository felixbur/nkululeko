# feats_praat.py
import os

import numpy as np
import pandas as pd

import nkululeko.glob_conf as glob_conf
from nkululeko.feat_extract import feinberg_praat
from nkululeko.feat_extract.featureset import Featureset


class PraatSet(Featureset):
    """A feature extractor for the Praat software.

    Based on David R. Feinberg's Praat scripts for the parselmouth python interface.
    https://osf.io/6dwr3/

    """

    def __init__(self, name, data_df, feats_type):
        super().__init__(name, data_df, feats_type)

    def extract(self):
        """Extract the features based on the initialized dataset or re-open them when found on disk."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = self.util.config_val("FEATS", "needs_feature_extraction", False)
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or no_reuse or not os.path.isfile(storage):
            self.util.debug("extracting Praat features, this might take a while...")
            self.df = feinberg_praat.compute_features(self.data_df.index)
            self.df = self.df.set_index(self.data_df.index)
            for i, col in enumerate(self.df.columns):
                if self.df[col].isnull().values.any():
                    self.util.debug(
                        f"{col} includes {self.df[col].isnull().sum()} nan,"
                        " inserting mean values"
                    )
                    self.df[col] = self.df[col].fillna(self.df[col].mean())

            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "false"
            except KeyError:
                pass
        else:
            self.util.debug(f"reusing extracted Praat features: {storage}.")
            self.df = self.util.get_store(storage, store_format)
        self.util.debug(f"praat feature names: {self.df.columns}")
        self.df = self.df.astype(float)

    def extract_sample(self, signal, sr):
        import audformat
        import audiofile

        tmp_audio_names = ["praat_audio_tmp.wav"]
        audiofile.write(tmp_audio_names[0], signal, sr)
        df = pd.DataFrame(index=tmp_audio_names)
        index = audformat.utils.to_segmented_index(df.index, allow_nat=False)
        df = feinberg_praat.compute_features(index)
        df.set_index(index)
        for i, col in enumerate(df.columns):
            if df[col].isnull().values.any():
                self.util.debug(
                    f"{col} includes {df[col].isnull().sum()} nan,"
                    " inserting mean values"
                )
                mean_val = df[col].mean()
                if not np.isnan(mean_val):
                    df[col] = df[col].fillna(mean_val)
                else:
                    df[col] = df[col].fillna(0)
        df = df.astype(float)
        feats = df.to_numpy()
        return feats
