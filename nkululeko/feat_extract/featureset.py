# featureset.py
import ast

import pandas as pd
from tqdm import tqdm

import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


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

    def extract(self):
        pass

    def _extract_embeddings_with_error_handling(self, extract_fn):
        """Process each file with extract_fn, skip failures, return filtered DataFrame.

        Args:
            extract_fn: callable(file, start, end) -> embedding array

        Returns:
            pd.DataFrame of embeddings with filtered index.
        """
        emb_series = pd.Series(index=self.data_df.index, dtype=object)
        for idx, (file, start, end) in enumerate(tqdm(self.data_df.index.to_list())):
            try:
                emb = extract_fn(file, start, end)
                emb_series.iloc[idx] = emb
            except Exception as e:
                self.util.warn(f"skipping {file}: {e}")
        valid = emb_series.notna()
        if not valid.all():
            self.util.warn(
                f"skipped {(~valid).sum()} files that failed to load or extract embeddings"
            )
            emb_series = emb_series[valid]
        return pd.DataFrame(emb_series.values.tolist(), index=emb_series.index)

    def filter(self):
        # use only the features that are indexed in the target dataframes
        self.df = self.df[self.df.index.isin(self.data_df.index)]
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
