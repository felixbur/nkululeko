"""Extract acoustic features from audio samples.

Extract acoustic features using several feature extractors 
(appends the features column-wise)
"""

import pandas as pd

from nkululeko.utils.util import Util


class FeatureExtractor:
    """Extract acoustic features from audio samples.

    Extract acoustic features using several feature extractors (appends the features column-wise).

    Args:
        data_df (pandas.DataFrame): dataframe with audiofile paths as index
        feats_types (List[str]): designations of acoustic feature extractors to be used
        data_name (str): name of databases that are extracted (for caching)
        feats_designation (str): the type of split (train/test), also is used for the cache name.

    Returns:
        df (pandas.DataFrame): dataframe with same index as data_df and acoustic features in columns
    """

    # pandas dataframe to store the features (and indexed with the data from the sets)
    df = None
    data_df = None  # dataframe to get audio paths

    def __init__(self, data_df, feats_types, data_name, feats_designation):
        self.data_df = data_df
        self.data_name = data_name
        self.feats_types = feats_types
        self.util = Util("feature_extractor")
        self.feats_designation = feats_designation

    def extract(self):
        self.feats = pd.DataFrame()
        for feats_type in self.feats_types:
            store_name = f"{self.data_name}_{feats_type}"
            self.feat_extractor = self._get_feat_extractor(store_name, feats_type)
            self.feat_extractor.extract()
            self.feat_extractor.filter()
            self.feats = pd.concat([self.feats, self.feat_extractor.df], axis=1)
        return self.feats

    def extract_sample(self, signal, sr):
        return self.feat_extractor.extract_sample(signal, sr)

    def _get_feat_extractor(self, store_name, feats_type):
        if isinstance(feats_type, list) and len(feats_type) == 1:
            feats_type = feats_type[0]
        feat_extractor_class = self._get_feat_extractor_class(feats_type)
        if feat_extractor_class is None:
            self.util.error(f"unknown feats_type: {feats_type}")
        return feat_extractor_class(
            f"{store_name}_{self.feats_designation}", self.data_df, feats_type
        )

    def _get_feat_extractor_class(self, feats_type):
        if feats_type == "os":
            from nkululeko.feat_extract.feats_opensmile import Opensmileset

            return Opensmileset

        elif feats_type == "spectra":
            from nkululeko.feat_extract.feats_spectra import Spectraloader

            return Spectraloader

        elif feats_type == "trill":
            from nkululeko.feat_extract.feats_trill import TRILLset

            return TRILLset

        elif feats_type.startswith(
            ("wav2vec2", "hubert", "wavlm", "spkrec", "whisper", "ast")
        ):
            return self._get_feat_extractor_by_prefix(feats_type)

        elif feats_type == "xbow":
            from nkululeko.feat_extract.feats_oxbow import Openxbow

            return Openxbow

        elif feats_type in (
            "audmodel",
            "auddim",
            "agender",
            "agender_agender",
            "snr",
            "mos",
            "squim",
            "clap",
            "praat",
            "mld",
            "import",
        ):
            return self._get_feat_extractor_by_name(feats_type)
        else:
            return None

    def _get_feat_extractor_by_prefix(self, feats_type):
        prefix, _, ext = feats_type.partition("-")
        from importlib import import_module

        module = import_module(f"nkululeko.feat_extract.feats_{prefix.lower()}")
        class_name = f"{prefix.capitalize()}"
        return getattr(module, class_name)

    def _get_feat_extractor_by_name(self, feats_type):
        from importlib import import_module

        module = import_module(f"nkululeko.feat_extract.feats_{feats_type.lower()}")
        class_name = f"{feats_type.capitalize()}Set"
        return getattr(module, class_name)
