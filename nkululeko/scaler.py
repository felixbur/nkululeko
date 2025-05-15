# scaler.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from nkululeko.utils.util import Util


class Scaler:
    """Class to normalize speech features."""

    def __init__(
        self,
        train_data_df,
        test_data_df,
        train_feats,
        test_feats,
        scaler_type,
        dev_x=None,
        dev_y=None,
    ):
        """Constructor.

        Parameters:
                train_data_df (pd.DataFrame): The training dataframe with speakers.
                    only needed for speaker normalization
                test_data_df (pd.DataFrame): The test dataframe with speakers
                    only needed for speaker normalization
                train_feats (pd.DataFrame): The train features dataframe
                test_feats (pd.DataFrame): The test features dataframe (can be None)
        """
        self.util = Util("scaler")
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "speaker":
            self.scaler = StandardScaler()
        elif scaler_type == "bins":
            pass
        else:
            self.util.error("unknown scaler: " + scaler_type)
        self.scaler_type = scaler_type
        self.feats_train = train_feats
        self.data_train = train_data_df
        self.feats_test = test_feats
        self.data_test = test_data_df
        if dev_x is not None:
            self.feats_dev = dev_y
            self.data_dev = dev_x
        else:
            self.feats_dev = None
            self.data_dev = None

    def scale(self):
        """Actually scales/normalizes.

        Returns:
                train_feats (pd.DataFrame): The scaled train features dataframe
                test_feats (pd.DataFrame): The scaled test features dataframe (can be None)
        """
        if self.scaler_type != "speaker":
            self.util.debug("scaling features based on training set")
            return self.scale_all()
        else:
            self.util.debug("scaling features per speaker based on training")
            return self.speaker_scale()

    def scale_all(self):
        if self.scaler_type != "bins":
            self.scaler.fit(self.feats_train.values)
            self.feats_train = self.scale_df(self.feats_train)
            if self.feats_test is not None:
                self.feats_test = self.scale_df(self.feats_test)
            if self.feats_dev is not None:
                self.feats_dev = self.scale_df(self.feats_dev)
        else:
            self.bin_to_three()
        if self.feats_dev is not None:
            return self.feats_train, self.feats_dev, self.feats_test
        else:
            return self.feats_train, self.feats_test

    def scale_df(self, df):
        scaled_features = self.scaler.transform(df.values)
        df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
        return df

    def speaker_scale(self):
        self.feats_train = self.speaker_scale_df(self.data_train, self.feats_train)
        if self.feats_test is not None:
            self.feats_test = self.speaker_scale_df(self.data_test, self.feats_test)
        if self.feats_dev is not None:
            self.feats_dev = self.speaker_scale_df(self.data_dev, self.feats_dev)
            return [self.feats_train, self.feats_dev, self.feats_test]
        else:
            return [self.feats_train, self.feats_test]

    def speaker_scale_df(self, df, feats_df):
        for speaker in df["speaker"].unique():
            indices = df.loc[df["speaker"] == speaker].index
            feats_df.loc[indices, :] = self.scaler.fit_transform(
                feats_df.loc[indices, :]
            )
        return feats_df

    def bin_to_three(self):
        feats_bin_train = pd.DataFrame(index=self.feats_train.index)
        feats_bin_test = pd.DataFrame(index=self.feats_test.index)
        for c in self.feats_train.columns:
            b1 = np.quantile(self.feats_train[c], 0.33)
            b2 = np.quantile(self.feats_train[c], 0.66)
            feats_bin_train[c] = self._bin(self.feats_train[c].values, b1, b2).values
            feats_bin_test[c] = self._bin(self.feats_test[c].values, b1, b2).values
        self.feats_train = feats_bin_train
        self.feats_test = feats_bin_test
        if self.feats_dev is not None:
            feats_bin_dev = pd.DataFrame(index=self.feats_dev.index)
            for c in self.feats_train.columns:
                b1 = np.quantile(self.feats_train[c], 0.33)
                b2 = np.quantile(self.feats_train[c], 0.66)
                feats_bin_dev[c] = self._bin(self.feats_dev[c].values, b1, b2).values
            self.feats_dev = feats_bin_dev

    def _bin(self, series, b1, b2):
        bins = [-1000000, b1, b2, 1000000]
        labels = [0, 0.5, 1]
        result = np.digitize(series, bins) - 1
        result = pd.Series(result)
        for index, lab in enumerate(labels):
            result = result.replace(index, str(lab))
        return result
