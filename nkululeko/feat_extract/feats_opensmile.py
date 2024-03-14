# opensmileset.py
from nkululeko.feat_extract.featureset import Featureset
import os
import pandas as pd
import nkululeko.glob_conf as glob_conf
import ast
import opensmile


class Opensmileset(Featureset):
    def __init__(self, name, data_df):
        super().__init__(name, data_df)
        self.featset = self.util.config_val("FEATS", "set", "eGeMAPSv02")
        try:
            self.feature_set = eval(f"opensmile.FeatureSet.{self.featset}")
            #'eGeMAPSv02, ComParE_2016, GeMAPSv01a, eGeMAPSv01a':
        except AttributeError:
            self.util.error(
                f"something is wrong with feature set: {self.featset}"
            )
        self.featlevel = self.util.config_val("FEATS", "level", "functionals")
        try:
            self.featlevel = self.featlevel.replace(
                "lld", "LowLevelDescriptors"
            )
            self.featlevel = self.featlevel.replace(
                "functionals", "Functionals"
            )
            self.feature_level = eval(
                f"opensmile.FeatureLevel.{self.featlevel}"
            )
        except AttributeError:
            self.util.error(
                f"something is wrong with feature level: {self.featlevel}"
            )

    def extract(self):
        """Extract the features based on the initialized dataset or re-open them when found on disk."""
        store = self.util.get_path("store")
        store_format = self.util.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{self.name}.{store_format}"
        extract = eval(
            self.util.config_val("FEATS", "needs_feature_extraction", "False")
        )
        no_reuse = eval(self.util.config_val("FEATS", "no_reuse", "False"))
        if extract or not os.path.isfile(storage) or no_reuse:
            self.util.debug(
                "extracting openSmile features, this might take a while..."
            )
            smile = opensmile.Smile(
                feature_set=self.feature_set,
                feature_level=self.feature_level,
                num_workers=5,
                verbose=True,
            )
            if isinstance(self.data_df.index, pd.MultiIndex):
                self.df = smile.process_index(self.data_df.index)
                self.df = self.df.set_index(self.data_df.index)
            else:
                self.df = smile.process_files(self.data_df.index)
                self.df.index = self.df.index.droplevel(1)
                self.df.index = self.df.index.droplevel(1)
            self.util.write_store(self.df, storage, store_format)
            try:
                glob_conf.config["DATA"]["needs_feature_extraction"] = "False"
            except KeyError:
                pass
        else:
            self.util.debug(f"reusing extracted OS features: {storage}.")
            self.df = self.util.get_store(storage, store_format)

    def extract_sample(self, signal, sr):
        smile = opensmile.Smile(
            feature_set=self.feature_set,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        feats = smile.process_signal(signal, sr)
        return feats.to_numpy()

    def filter(self):
        # use only the features that are indexed in the target dataframes
        self.df = self.df[self.df.index.isin(self.data_df.index)]
        try:
            # use only some features
            selected_features = ast.literal_eval(
                glob_conf.config["FEATS"]["os.features"]
            )
            self.util.debug(
                f"selecting features from opensmile: {selected_features}"
            )
            sel_feats_df = pd.DataFrame()
            hit = False
            for feat in selected_features:
                try:
                    sel_feats_df[feat] = self.df[feat]
                    hit = True
                except KeyError:
                    pass
            if hit:
                self.df = sel_feats_df
                self.util.debug(
                    "new feats shape after selecting opensmile features:"
                    f" {self.df.shape}"
                )
        except KeyError:
            pass
