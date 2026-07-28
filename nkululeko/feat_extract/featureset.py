# featureset.py
import ast

import pandas as pd

import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util

# Exceptions expected from per-file feature extraction (bad/corrupt audio,
# unsupported sample rate, missing files, etc). Anything outside this set
# (e.g. KeyboardInterrupt, SystemExit) is allowed to propagate.
EXTRACTION_ERRORS = (IOError, OSError, RuntimeError, ValueError, AssertionError)


class Featureset:
    name = ""  # designation
    df = None  # pandas dataframe to store the features
    # (and indexed with the data from the sets)
    data_df = None  # dataframe to get audio paths

    def __init__(self, name, data_df, feats_type):
        """Constructor.

        Args:
        name (str): The name of the feature set.
        data_df (pd.DataFrame): The dataframe containing the data to extract features from.
        feats_type (str): The type of features to extract.
        """
        self.name = name
        self.data_df = data_df
        self.util = Util("featureset")
        self.feats_type = feats_type
        self.n_jobs = int(self.util.config_val("MODEL", "n_jobs", "8"))

    def _needs_extraction(self, storage):
        """Check whether features need to be (re-)extracted.

        Args:
            storage: Path to the stored features file.

        Returns:
            bool: True if extraction is needed.
        """
        import os

        extract = self.util.config_val_bool("FEATS", "needs_feature_extraction", False)
        no_reuse = self.util.config_val_bool("FEATS", "no_reuse", False)
        return no_reuse or extract or not os.path.isfile(storage)

    def extract(self):
        pass

    def _get_fail_threshold(self, default=0.5):
        """Read FEATS.fail_threshold from config, validated and clamped to [0.0, 1.0].

        Falls back to `default` (with a warning) if the config value is not a
        valid number, so a malformed config entry can't crash extraction
        before it starts.
        """
        raw = self.util.config_val("FEATS", "fail_threshold", str(default))
        try:
            threshold = float(raw)
        except (TypeError, ValueError):
            self.util.warn(
                f"invalid FEATS.fail_threshold {raw!r}, using default {default}"
            )
            return default
        if not 0.0 <= threshold <= 1.0:
            self.util.warn(
                f"FEATS.fail_threshold {threshold} out of range [0.0, 1.0],"
                f" clamping"
            )
            threshold = min(max(threshold, 0.0), 1.0)
        return threshold

    def _extract_embeddings_with_error_handling(self, extract_fn):
        """Process each file with extract_fn, skip failures, return filtered DataFrame.

        Args:
            extract_fn: callable(file, start, end) -> embedding array

        Returns:
            pd.DataFrame of embeddings with filtered index.
        """
        emb_series = pd.Series(index=self.data_df.index, dtype=object)
        iterable = self.data_df.index.to_list()
        total = len(iterable)
        failed = 0
        fail_threshold = self._get_fail_threshold()
        try:
            # Use tqdm for a progress bar if available, but don't require it.
            from tqdm import tqdm  # type: ignore[import-not-found]

            iterable = tqdm(iterable)
        except ImportError:
            # Fall back to plain iteration without a progress bar.
            pass

        for idx, (file, start, end) in enumerate(iterable):
            try:
                emb = extract_fn(file, start, end)
                emb_series.iloc[idx] = emb
            except EXTRACTION_ERRORS as e:
                self.util.warn(f"skipping {file}: {e}")
                failed += 1

        if failed > 0:
            self.util.warn(
                f"Feature extraction: {failed}/{total} files failed"
                f" ({100 * failed / total:.1f}%)"
            )
            if total > 0 and failed / total > fail_threshold:
                self.util.error(
                    f"Extraction failure rate {failed / total:.0%} exceeds"
                    f" threshold {fail_threshold:.0%}"
                )

        valid = emb_series.notna()
        if not valid.all():
            emb_series = emb_series[valid]
        return pd.DataFrame(emb_series.values.tolist(), index=emb_series.index)

    def filter(self):
        # use only the features that are indexed in the target dataframes
        self.df = self.util.filter_filepath(self.data_df, self.df)
        try:
            # use only some features
            selected_features = ast.literal_eval(glob_conf.config["FEATS"]["features"])
            self.util.debug(f"selecting features: {selected_features}")
            sel_feats_df = pd.DataFrame()
            hit = False
            for feat in selected_features:
                try:
                    sel_feats_df[feat] = self.df[feat]
                    hit = True
                except KeyError:
                    self.util.warn(f"non existent feature in {self.feats_type}: {feat}")
                    pass
            if hit:
                self.df = sel_feats_df
                self.util.debug(
                    f"new feats shape after selecting features for {self.feats_type}: {self.df.shape}"
                )
        except KeyError:
            pass
