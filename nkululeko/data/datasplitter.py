# datasplitter.py
import ast
import os
import os.path

import pandas as pd
import secrets
from sklearn.preprocessing import LabelEncoder
import pickle
from nkululeko.filter_data import DataFilter
import nkululeko.glob_conf as glob_conf
from nkululeko.file_checker import FileChecker
from nkululeko.utils.util import Util


class Datasplitter:
    def __init__(self, datasets):
        self.util = Util("datasplitter")
        self.datasets = datasets
        self.split3 = glob_conf.split3
        self.target = glob_conf.target
        self.got_speaker = glob_conf.got_speaker

    def get_sample_selection(self) -> pd.DataFrame:
        """Get the dataframe based on the sample selection configuration.

        Returns:
            pd.DataFrame: The selected dataframe based on the configuration.
        """
        sample_selection = self.util.config_val("EXP", "sample_selection", "all")
        if sample_selection == "all":
            df = pd.concat([self.df_train, self.df_test])
        elif sample_selection == "train":
            df = self.df_train
        elif sample_selection == "test":
            df = self.df_test
        else:
            self.util.error(
                f"unknown selection specifier {sample_selection},"
                " should be [all | train | test]"
            )
        return df

    def _add_random_target(self, df):
        labels = glob_conf.labels
        a = [None] * len(df)
        for i in range(0, len(df)):
            a[i] = secrets.choice(labels)
        df[self.target] = a
        return df

    def fill_train_and_tests(self):
        """Set up train and development sets. The method should be specified in the config."""
        self.df_train, self.df_test, self.df_dev = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        if self.split3:
            self.df_dev = pd.DataFrame()
        else:
            self.df_dev = None
        for d in self.datasets.values():
            if self.split3:
                d.split_3()
            else:
                d.split()
            # Prepare labels only for supervised experiments
            if self.target is not None and self.target != "none":
                d.prepare_labels()
            if d.df_train.shape[0] == 0:
                self.util.debug(f"warn: {d.name} train empty")
            else:
                self.df_train = pd.concat([self.df_train, d.df_train])
                self.util.copy_flags(d, self.df_train)
            if d.df_test.shape[0] == 0:
                self.util.debug(f"warn: {d.name} test empty")
            else:
                self.df_test = pd.concat([self.df_test, d.df_test])
                self.util.copy_flags(d, self.df_test)
            if self.split3:
                if d.df_dev.shape[0] == 0:
                    self.util.debug(f"warn: {d.name} dev empty")
                else:
                    self.df_dev = pd.concat([self.df_dev, d.df_dev])
                    self.util.copy_flags(d, self.df_dev)

        # Return early for unlabeled/unsupervised runs, but still return the split dataframes
        if self.target is None or self.target == "none":
            if self.split3:
                return self.df_train, self.df_test, self.df_dev
            return self.df_train, self.df_test
        self.util.copy_flags(self, self.df_test)
        self.util.copy_flags(self, self.df_train)
        if self.split3:
            self.util.copy_flags(self, self.df_dev)
        # Try data checks
        datachecker = FileChecker(self.df_train)
        self.df_train = datachecker.all_checks()
        datachecker.set_data(self.df_test)
        self.df_test = datachecker.all_checks()
        if self.split3:
            datachecker.set_data(self.df_dev)
            self.df_dev = datachecker.all_checks()

        # Check for filters
        filter_sample_selection = self.util.config_val(
            "EXP", "filter.sample_selection", "all"
        )
        if filter_sample_selection == "all":
            datafilter = DataFilter(self.df_train)
            self.df_train = datafilter.all_filters()
            datafilter = DataFilter(self.df_test)
            self.df_test = datafilter.all_filters()
            if self.split3:
                datafilter = DataFilter(self.df_dev)
                self.df_dev = datafilter.all_filters()
        elif filter_sample_selection == "train":
            datafilter = DataFilter(self.df_train)
            self.df_train = datafilter.all_filters()
        elif filter_sample_selection == "test":
            datafilter = DataFilter(self.df_test)
            self.df_test = datafilter.all_filters()
        else:
            msg = (
                "unkown filter sample selection specifier"
                f" {filter_sample_selection}, should be [all | train | test]"
            )
            self.util.error(msg)

        # encode the labels
        if self.util.exp_is_classification():
            datatype = self.util.config_val("DATA", "type", "dummy")
            if datatype == "continuous":
                if not self.df_test.empty:
                    test_cats = self.df_test["class_label"].unique()
                if not self.df_train.empty:
                    train_cats = self.df_train["class_label"].unique()
                if self.split3 and not self.df_dev.empty:
                    dev_cats = self.df_dev["class_label"].unique()
            else:
                if not self.df_test.empty:
                    if self.df_test.is_labeled:
                        # get printable string of categories and their counts
                        test_cats = self.df_test[self.target].value_counts().to_string()
                    else:
                        # if there is no target, copy a dummy label
                        self.df_test = self._add_random_target(self.df_test).astype(
                            "str"
                        )
                if not self.df_train.empty:
                    train_cats = self.df_train[self.target].value_counts().to_string()
                if self.split3 and not self.df_dev.empty:
                    dev_cats = self.df_dev[self.target].value_counts().to_string()
            # encode the labels as numbers
            self.label_encoder = LabelEncoder()
            glob_conf.set_label_encoder(self.label_encoder)
            if not self.df_train.empty:
                self.util.debug(f"Categories train: {train_cats}")
                self.df_train[self.target] = self.label_encoder.fit_transform(
                    self.df_train[self.target]
                )
            if not self.df_test.empty:
                if self.df_test.is_labeled:
                    self.util.debug(f"Categories test: {test_cats}")
                if not self.df_train.empty:
                    self.df_test[self.target] = self.label_encoder.transform(
                        self.df_test[self.target]
                    )
            if self.split3 and not self.df_dev.empty:
                self.util.debug(f"Categories dev: {dev_cats}")
                if not self.df_train.empty:
                    self.df_dev[self.target] = self.label_encoder.transform(
                        self.df_dev[self.target]
                    )
        if self.got_speaker:
            speakers_train = (
                0
                if self.df_train.empty or "speaker" not in self.df_train.columns
                else self.df_train.speaker.nunique()
            )
            speakers_test = (
                0
                if self.df_test.empty or "speaker" not in self.df_test.columns
                else self.df_test.speaker.nunique()
            )
            self.util.debug(
                f"{speakers_test} speakers in test and"
                f" {speakers_train} speakers in train"
            )
            if self.split3:
                speakers_dev = (
                    0
                    if self.df_dev.empty or "speaker" not in self.df_dev.columns
                    else self.df_dev.speaker.nunique()
                )
                self.util.debug(f"{speakers_dev} speakers in dev")

        target_factor = self.util.config_val("DATA", "target_divide_by", False)
        if target_factor:
            self.df_test[self.target] = self.df_test[self.target] / float(target_factor)
            self.df_train[self.target] = self.df_train[self.target] / float(
                target_factor
            )
            if self.split3:
                self.df_dev[self.target] = self.df_dev[self.target] / float(
                    target_factor
                )
            if not self.util.exp_is_classification():
                self.df_test["class_label"] = self.df_test["class_label"] / float(
                    target_factor
                )
                self.df_train["class_label"] = self.df_train["class_label"] / float(
                    target_factor
                )
                if self.split3:
                    self.df_dev["class_label"] = self.df_dev["class_label"] / float(
                        target_factor
                    )
        if self.split3:
            shapes = f"{self.df_train.shape}/{self.df_dev.shape}/{self.df_test.shape}"
            self.util.debug(f"train/dev/test shape: {shapes}")
            if self.got_speaker and "speaker" in self.df_dev.columns:
                dev_spkrs = list(map(str, self.df_dev.speaker.unique()))
                self.util.debug(f"dev speakers: {dev_spkrs}")
        else:
            self.util.debug(
                f"train/test shape: {self.df_train.shape}/{self.df_test.shape}"
            )
        if not self.df_train.empty and "speaker" in self.df_train.columns:
            train_spkrs = list(map(str, self.df_train.speaker.unique()))
            self.util.debug(f"train speakers: {train_spkrs}")
        if not self.df_test.empty and "speaker" in self.df_test.columns:
            test_spkrs = list(map(str, self.df_test.speaker.unique()))
            self.util.debug(f"test speakers: {test_spkrs}")

        # Build per-dataset test mapping for multi-test-set evaluation
        self._build_test_ds_df()

        if self.split3:
            return self.df_train, self.df_test, self.df_dev
        else:
            return self.df_train, self.df_test

    def _build_test_ds_df(self):
        """Build a dict mapping dataset name to its portion of the (encoded) test set.

        This enables per-dataset evaluation when multiple datasets contribute test
        samples. The mapping is built from the in-memory dataset splits
        (``dataset.df_test``) and falls back to legacy ``{name}_testdf.pkl``
        caches if needed.
        """
        self.test_ds_df = {}
        if self.df_test.empty:
            return
        store_path = self.util.get_path("store")
        for name, dataset in self.datasets.items():
            ds_test_source = getattr(dataset, "df_test", None)
            if ds_test_source is not None and ds_test_source.shape[0] > 0:
                ds_test = self.df_test[self.df_test.index.isin(ds_test_source.index)]
                if ds_test.shape[0] > 0:
                    self.test_ds_df[name] = ds_test
                continue
            storage_test = f"{store_path}{name}_testdf.pkl"
            if os.path.isfile(storage_test):
                try:
                    ds_test_cached = pd.read_pickle(storage_test)
                    if ds_test_cached.shape[0] > 0:
                        ds_test = self.df_test[
                            self.df_test.index.isin(ds_test_cached.index)
                        ]
                        if ds_test.shape[0] > 0:
                            self.test_ds_df[name] = ds_test
                except (pickle.UnpicklingError, ValueError, AttributeError) as e:
                    self.util.warn(f"{name}: could not load split cache: {e}")

    def extract_feats_for_datasets(self):
        all_feats = pd.DataFrame()
        feats_types = self.util.config_val("FEATS", "type", "os")
        if isinstance(feats_types, str):
            if feats_types.startswith("[") and feats_types.endswith("]"):
                feats_types = ast.literal_eval(feats_types)
            else:
                feats_types = [feats_types]
        if len(feats_types) == 0:
            return all_feats
        for d in self.datasets.values():
            d_feats, self.feature_extractor = d.extract_features(feats_types)
            all_feats = pd.concat([all_feats, d_feats])
        self.util.debug(f"All dataset features shape: {all_feats.shape}")
        return all_feats

    def extract_feats(self):
        """Extract the features for train and dev sets.

        They will be stored on disk and need to be removed manually.

        The string FEATS.feats_type is read from the config, defaults to os.

        """
        all_feats = self.extract_feats_for_datasets()
        self.feats_test, self.feats_train = pd.DataFrame(), pd.DataFrame()
        if self.split3:
            self.feats_dev = pd.DataFrame()
        else:
            self.feats_dev = None
        # for some models no features are needed
        if all_feats.empty:
            self.util.debug("no feature extractor specified.")
            if self.split3:
                return None, None, None
            else:
                return None, None
        if not self.df_train.empty:
            self.feats_train = self.util.filter_filepath(self.df_train, all_feats)
            # self.feats_train = all_feats[all_feats.index.isin(self.df_train.index)]
        if not self.df_test.empty:
            self.feats_test = self.util.filter_filepath(self.df_test, all_feats)
            # self.feats_test = all_feats[all_feats.index.isin(self.df_test.index)]
        if self.split3:
            if not self.df_dev.empty:
                self.feats_dev = self.util.filter_filepath(self.df_dev, all_feats)
                # self.feats_dev = all_feats[all_feats.index.isin(self.df_dev.index)]
                shps = f"{self.feats_train.shape}/{self.feats_dev.shape}/{self.feats_test.shape}"
                self.util.debug(f"Train/dev/test features:{shps}")
        else:
            self.util.debug(
                f"All features: train shape : {self.feats_train.shape}, test"
                f" shape:{self.feats_test.shape}"
            )
        if self.feats_train.shape[0] < self.df_train.shape[0]:
            self.util.warn(
                f"train feats ({self.feats_train.shape[0]}) != train labels"
                f" ({self.df_train.shape[0]})"
            )
            self.df_train = self.util.filter_filepath(self.feats_train, self.df_train)
            # self.df_train = self.df_train[
            #     self.df_train.index.isin(self.feats_train.index)
            # ]
            self.util.warn(f"new train labels shape: {self.df_train.shape[0]}")
        if self.feats_test.shape[0] < self.df_test.shape[0]:
            self.util.warn(
                f"test feats ({self.feats_test.shape[0]}) != test labels"
                f" ({self.df_test.shape[0]})"
            )
            self.df_test = self.util.filter_filepath(self.feats_test, self.df_test)
            # self.df_test = self.df_test[self.df_test.index.isin(self.feats_test.index)]
            self.util.warn(f"new test labels shape: {self.df_test.shape[0]}")
        if self.split3:
            if self.feats_dev.shape[0] < self.df_dev.shape[0]:
                self.util.warn(
                    f"dev feats ({self.feats_dev.shape[0]}) != dev labels"
                    f" ({self.df_dev.shape[0]})"
                )
                self.df_dev = self.util.filter_filepath(self.feats_dev, self.df_dev)
                # self.df_dev = self.df_dev[self.df_dev.index.isin(self.feats_dev.index)]
                self.util.warn(f"new dev labels shape: {self.df_dev.shape[0]}")

            return self.feats_train, self.feats_test, self.feats_dev
        else:
            return self.feats_train, self.feats_test
