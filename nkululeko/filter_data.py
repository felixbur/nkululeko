import audformat
import pandas as pd
import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util
import ast


class DataFilter:
    def __init__(self, df):
        self.util = Util("datafilter")
        self.df = df.copy()
        self.util.copy_flags(df, self.df)

    def all_filters(self, data_name=""):
        self.limit_samples(data_name)
        self.limit_speakers(data_name)
        self.filter_value(data_name)
        return self.filter_duration(data_name)

    def limit_samples(self, data_name=""):
        """limit number of samples
        the samples are selected randomly
        """
        if data_name == "":
            max = self.util.config_val("DATA", "limit_samples", False)
        else:
            max = self.util.config_val_data(data_name, "limit_samples", False)
        if max:
            if self.df.shape[0] < int(max):
                return self.df
            else:
                df = self.df.sample(int(max))
                self.util.debug(
                    f"{data_name}: limited samples to {max}, reduced samples"
                    f" from {self.df.shape[0]} to {df.shape[0]}"
                )
                self.util.copy_flags(self.df, df)
                self.df = df
                return df
        else:
            return self.df

    def limit_speakers(self, data_name=""):
        """limit number of samples per speaker
        the samples are selected randomly
        """
        if data_name == "":
            max = self.util.config_val("DATA", "limit_samples_per_speaker", False)
        else:
            max = self.util.config_val_data(
                data_name, "limit_samples_per_speaker", False
            )
        if max:
            df = pd.DataFrame()
            for s in self.df.speaker.unique():
                s_df = self.df[self.df["speaker"].eq(s)]
                if s_df.shape[0] < int(max):
                    df = pd.concat([df, s_df])
                else:
                    df = pd.concat([df, s_df.sample(int(max))])
            self.util.debug(
                f"{data_name}: limited samples to {max} per speaker, reduced"
                f" samples from {self.df.shape[0]} to {df.shape[0]}"
            )
            self.util.copy_flags(self.df, df)
            self.df = df
            return df
        else:
            return self.df

    def filter_duration(self, data_name=""):
        """remove all samples less than min_dur duration"""
        if data_name == "":
            min_dur = self.util.config_val("DATA", "min_duration_of_sample", False)
            max_dur = self.util.config_val("DATA", "max_duration_of_sample", False)
        else:
            min_dur = self.util.config_val_data(
                data_name, "min_duration_of_sample", False
            )
            max_dur = self.util.config_val_data(
                data_name, "max_duration_of_sample", False
            )
        if min_dur or max_dur:
            if not isinstance(self.df.index, pd.MultiIndex):
                self.util.debug(
                    "converting file index to multi index, this might take a"
                    " while..."
                )
                self.df.index = audformat.utils.to_segmented_index(
                    self.df.index, allow_nat=False
                )
            if min_dur:
                old_samples = self.df.shape[0]
                df = self.df.copy()
                for i in self.df.index:
                    start = i[1]
                    end = i[2]
                    dur = (end - start).total_seconds()
                    if dur < float(min_dur):
                        df = df.drop(i, axis=0)
                self.util.debug(
                    f"{data_name}: filtered samples less than"
                    f" {min_dur} seconds, reduced samples from {old_samples} to"
                    f" {df.shape[0]}"
                )
            if max_dur:
                old_samples = self.df.shape[0]
                df = self.df.copy()
                for i in self.df.index:
                    start = i[1]
                    end = i[2]
                    dur = (end - start).total_seconds()
                    if dur > float(max_dur):
                        df = df.drop(i, axis=0)
                self.util.debug(
                    f"{data_name}: filtered samples more than"
                    f" {max_dur} seconds, reduced samples from {old_samples} to"
                    f" {df.shape[0]}"
                )
            self.util.copy_flags(self.df, df)
            self.df = df
            return df
        else:
            return self.df

    def filter_value(self, data_name=""):
        if data_name == "":
            filters_str = self.util.config_val("DATA", "filter", False)
        else:
            filters_str = self.util.config_val_data(data_name, "filter", False)
        if filters_str:
            filters = ast.literal_eval(filters_str)
            df = self.df.copy()
            for f in filters:
                col = f[0]
                val = f[1]
                pre = df.shape[0]
                df = df[df[col] == val]
                post = df.shape[0]
                self.util.debug(
                    f"{data_name}: filtered {col}={val}, reduced samples from"
                    f" {pre} to {post}"
                )
            self.util.copy_flags(self.df, df)
            self.df = df
            return df
        else:
            return self.df


def limit_speakers(df, max=20):
    """limit number of samples per speaker
    the samples are selected randomly
    """
    df_ret = pd.DataFrame()
    for s in df.speaker.unique():
        s_df = df[df["speaker"].eq(s)]
        if s_df.shape[0] < max:
            df_ret = df_ret.append(s_df)
        else:
            df_ret = df_ret.append(s_df.sample(max))
    return df_ret


def filter_min_dur(df, min_dur):
    """remove all samples less than min_dur duration"""
    df_ret = df.copy()
    if not isinstance(df.index, pd.MultiIndex):
        glob_conf.util.debug(
            "converting file index to multi index, this might take a while..."
        )
        df_ret.index = audformat.utils.to_segmented_index(df.index, allow_nat=False)
    for i in df_ret.index:
        start = i[1]
        end = i[2]
        dur = (end - start).total_seconds()
        if dur < float(min_dur):
            df_ret = df_ret.drop(i, axis=0)
    df_ret.is_labeled = df.is_labeled
    df_ret.got_gender = df.got_gender
    df_ret.got_speaker = df.got_speaker
    return df_ret


def filter_max_dur(df, max_dur):
    """remove all samples less than min_dur duration"""
    df_ret = df.copy()
    if not isinstance(df.index, pd.MultiIndex):
        glob_conf.util.debug(
            "converting file index to multi index, this might take a while..."
        )
        df_ret.index = audformat.utils.to_segmented_index(df.index, allow_nat=False)
    for i in df_ret.index:
        start = i[1]
        end = i[2]
        dur = (end - start).total_seconds()
        if dur > float(max_dur):
            df_ret = df_ret.drop(i, axis=0)
    df_ret.is_labeled = df.is_labeled
    df_ret.got_gender = df.got_gender
    df_ret.got_speaker = df.got_speaker
    return df_ret
