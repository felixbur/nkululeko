"""Extract acoustic features from audio samples.

Extract acoustic features using several feature extractors
(appends the features column-wise)
"""

import numpy as np
import pandas as pd

from nkululeko.experiment_context import ContextAware, use_context
from nkululeko.utils.sequences import is_sequence, pool, pooling_stats
from nkululeko.utils.util import Util


class FeatureExtractor(ContextAware):
    """Extract acoustic features from audio samples.

    Extract acoustic features using several feature extractors (appends the features column-wise).

    Args:
        data_df (pandas.DataFrame): dataframe with audiofile paths as index
        feats_types (List[str]): designations of acoustic feature extractors to be used
        data_name (str): name of databases that are extracted (for caching)
        feats_designation (str): the type of split (train/test), also is used for the cache name.

    Returns:
        df (pandas.DataFrame): dataframe with same index as data_df and acoustic features in columns;
        with FEATS.level = frame and a sequence model (see takes_sequences()), one row per frame,
        index (file, start, end, frame), otherwise frame sequences are pooled per sample
    """

    # pandas dataframe to store the features (and indexed with the data from the sets)
    df = None
    data_df = None  # dataframe to get audio paths

    def __init__(
        self, data_df, feats_types, data_name, feats_designation, context=None
    ):
        self.data_df = data_df
        self.data_name = data_name
        self.feats_types = feats_types
        self.util = Util("feature_extractor", context=context)
        self.context = self.util.context
        self.feats_designation = feats_designation

    def extract(self):
        """Extract and column-wise concatenate features for every configured feats_type.

        Returns:
            pandas.DataFrame: features for all samples in ``data_df``, indexed the
            same way, with one block of columns per feats_type in ``feats_types``.
        """
        parts = []
        for feats_type in self.feats_types:
            store_name = f"{self.data_name}_{feats_type}"
            self.feat_extractor = self._get_feat_extractor(store_name, feats_type)
            self.feat_extractor.extract()
            self.feat_extractor.filter()
            df = self.feat_extractor.df
            if is_sequence(df) and not self.takes_sequences():
                df = pool(df, self.pooling_stats())
            parts.append(df)
        self.feats = self._join(parts)
        return self.feats

    def takes_sequences(self):
        """True if the configured model takes frame sequences; for every
        other model, frame sequences are pooled to one vector per sample."""
        from nkululeko.utils.naming import SEQUENCE_MODEL_TYPES

        return self.util.config_val("MODEL", "type", "") in SEQUENCE_MODEL_TYPES

    def pooling_stats(self):
        return pooling_stats(self.util)

    def _join(self, parts):
        """Join the feature types column-wise. Frame sequences can only be
        joined with frame sequences on the same frames."""
        if not parts:
            return pd.DataFrame()
        sequences = [is_sequence(df) for df in parts]
        if any(sequences) and not all(sequences):
            self.util.error(
                "FEATS.type mixes frame sequences with one-vector-per-sample "
                "features, which a sequence model can't take; use feature "
                "types with frame output only (FEATS.level = frame)"
            )
        if any(sequences) and len(parts) > 1:
            first = parts[0].index
            if not all(df.index.equals(first) for df in parts[1:]):
                self.util.error(
                    "FEATS.type: the frame sequences of the feature types "
                    "have different frames and can't be joined"
                )
        return pd.concat(parts, axis=1)

    def extract_sample(self, signal, sr):
        """Extract features for a single in-memory audio sample.

        Delegates to the underlying feature extractor selected by ``extract()``
        (``self.feat_extractor``), so ``extract()`` must be called at least once
        before this method is used. If ``feats_types`` has more than one
        entry, ``self.feat_extractor`` is whichever extractor ran last in
        ``extract()``'s loop, so only that one feats_type's features are
        returned here - unlike ``extract()`` itself, which concatenates all
        of them.

        Args:
            signal (numpy.ndarray): Audio signal as a numpy array.
            sr (int): Sample rate of the audio signal.

        Returns:
            numpy.ndarray: Extracted features for the sample, from the last
            feats_type processed by ``extract()``.
        """
        feats = self.feat_extractor.extract_sample(signal, sr)
        if self.feat_extractor.wants_frames() and not self.takes_sequences():
            # the same pooling as in extract(): one block per statistic
            frames = np.atleast_2d(np.asarray(feats, dtype=float))
            stat_funcs = {
                "mean": np.mean,
                "std": np.std,  # population std, as in pool()
                "min": np.min,
                "max": np.max,
                "median": np.median,
            }
            feats = np.concatenate(
                [stat_funcs[s](frames, axis=0) for s in self.pooling_stats()]
            )[np.newaxis, :]
        return feats

    def _get_feat_extractor(self, store_name, feats_type):
        if isinstance(feats_type, list) and len(feats_type) == 1:
            feats_type = feats_type[0]
        feat_extractor_class = self._get_feat_extractor_class(feats_type)
        if feat_extractor_class is None:
            self.util.error(f"unknown feats_type: {feats_type}")
        with use_context(self.context):
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

        elif feats_type == "mld":
            from nkululeko.feat_extract.feats_mld import MLD_set

            return MLD_set

        elif feats_type.startswith(
            (
                "bert",
                "wav2vec2",
                "hubert",
                "wavlm",
                "spkrec",
                "whisper",
                "ast",
                "emotion2vec",
            )
        ):
            return self._get_feat_extractor_by_prefix(feats_type)

        elif feats_type == "xbow":
            from nkululeko.feat_extract.feats_oxbow import Openxbow

            return Openxbow

        elif feats_type in (
            "audmodel",
            "audwav2vec2",
            "auddim",
            "agender",
            "agender_agender",
            "snr",
            "mos",
            "squim",
            "clap",
            "praat",
            "voicesauce",
            "mld",
            "import",
            "sptk",
            "lfcc",
            "cqcc",
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
