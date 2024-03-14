"""
feature_extractor.py

Helper class to encapsulate feature extraction methods

"""
import pandas as pd

from nkululeko.utils.util import Util


class FeatureExtractor:
    """
    Extract acoustic features from audio samples, using several feature extractors (appends the features column-wise)
    Args:
        data_df (pandas.DataFrame): dataframe with audiofile paths as index
        feats_types (array of strings): designations of acoustic feature extractors to be used
        data_name (string): names of databases that are extracted (for the caching)
        feats_designation (string): the type of split (train/test), also is used for the cache name.
    Returns:
        df (pandas.DataFrame): dataframe with same index as data_df and acoustic features in columns
    """

    # pandas dataframe to store the features (and indexed with the data from the sets)
    df = None
    data_df = None  # dataframe to get audio paths

    # def __init__
    def __init__(self, data_df, feats_types, data_name, feats_designation):
        self.data_df = data_df
        self.data_name = data_name
        self.feats_types = feats_types
        self.util = Util("feature_extractor")
        self.feats_designation = feats_designation

    def extract(self):
        # feats_types = self.util.config_val_list('FEATS', 'type', ['os'])
        self.featExtractor = None
        self.feats = pd.DataFrame()
        _scale = True
        for feats_type in self.feats_types:
            store_name = f"{self.data_name}_{feats_type}"
            if feats_type == "os":
                from nkululeko.feat_extract.feats_opensmile import Opensmileset

                self.featExtractor = Opensmileset(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "spectra":
                from nkululeko.feat_extract.feats_spectra import Spectraloader

                self.featExtractor = Spectraloader(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "trill":
                from nkululeko.feat_extract.feats_trill import TRILLset

                self.featExtractor = TRILLset(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type.startswith("wav2vec"):
                from nkululeko.feat_extract.feats_wav2vec2 import Wav2vec2

                self.featExtractor = Wav2vec2(
                    f"{store_name}_{self.feats_designation}",
                    self.data_df,
                    feats_type,
                )
            elif feats_type.startswith("hubert"):
                from nkululeko.feat_extract.feats_hubert import Hubert

                self.featExtractor = Hubert(
                    f"{store_name}_{self.feats_designation}",
                    self.data_df,
                    feats_type,
                )

            elif feats_type.startswith("wavlm"):
                from nkululeko.feat_extract.feats_wavlm import Wavlm

                self.featExtractor = Wavlm(
                    f"{store_name}_{self.feats_designation}",
                    self.data_df,
                    feats_type,
                )

            elif feats_type.startswith("spkrec"):
                from nkululeko.feat_extract.feats_spkrec import Spkrec

                self.featExtractor = Spkrec(
                    f"{store_name}_{self.feats_designation}",
                    self.data_df,
                    feats_type,
                )
            elif feats_type == "audmodel":
                from nkululeko.feat_extract.feats_audmodel import AudModelSet

                self.featExtractor = AudModelSet(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "auddim":
                from nkululeko.feat_extract.feats_audmodel_dim import (
                    AudModelDimSet,
                )

                self.featExtractor = AudModelDimSet(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "agender":
                from nkululeko.feat_extract.feats_agender import (
                    AudModelAgenderSet,
                )

                self.featExtractor = AudModelAgenderSet(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "agender_agender":
                from nkululeko.feat_extract.feats_agender_agender import (
                    AgenderAgenderSet,
                )

                self.featExtractor = AgenderAgenderSet(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "snr":
                from nkululeko.feat_extract.feats_snr import SNRSet

                self.featExtractor = SNRSet(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "mos":
                from nkululeko.feat_extract.feats_mos import MOSSet

                self.featExtractor = MOSSet(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "squim":
                from nkululeko.feat_extract.feats_squim import SQUIMSet

                self.featExtractor = SQUIMSet(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "clap":
                from nkululeko.feat_extract.feats_clap import Clap

                self.featExtractor = Clap(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "praat":
                from nkululeko.feat_extract.feats_praat import Praatset

                self.featExtractor = Praatset(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "mld":
                from nkululeko.feat_extract.feats_mld import MLD_set

                self.featExtractor = MLD_set(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            elif feats_type == "import":
                from nkululeko.feat_extract.feats_import import Importset

                self.featExtractor = Importset(
                    f"{store_name}_{self.feats_designation}", self.data_df
                )
            else:
                self.util.error(f"unknown feats_type: {feats_type}")

            self.featExtractor.extract()
            self.featExtractor.filter()
            # remove samples that were not extracted by MLD
            # self.df_test = self.df_test.loc[self.df_test.index.intersection(featExtractor_test.df.index)]
            # self.df_train = self.df_train.loc[self.df_train.index.intersection(featExtractor_train.df.index)]
            self.util.debug(f"{feats_type}: shape : {self.featExtractor.df.shape}")
            self.feats = pd.concat([self.feats, self.featExtractor.df], axis=1)
        return self.feats

    def extract_sample(self, signal, sr):
        return self.featExtractor.extract_sample(signal, sr)
