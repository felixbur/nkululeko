# dataframe.py - mixin for DataFrame, label, and numeric helpers
import ast

import audformat
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


class DataFrameMixin:
    """Mixin providing DataFrame manipulation and label encoding methods for Util."""

    def is_categorical(self, pd_series):
        """Check if a dataframe column is categorical."""
        return (
            pd_series.dtype.name == "object"
            or pd_series.dtype.name == "bool"
            or pd_series.dtype.name == "string"
            or isinstance(pd_series.dtype, pd.CategoricalDtype)
            or isinstance(pd_series.dtype, pd.BooleanDtype)
            or isinstance(pd_series.dtype, pd.StringDtype)
        )

    def is_numeric(self, pd_series):
        """Check if a dataframe column is numeric.

        Uses pandas.api.types.is_numeric_dtype to properly handle all numeric dtypes
        including int32, float32, int64, float64, and nullable integer/float types.
        """
        return is_numeric_dtype(pd_series)

    def make_segmented_index(self, df):
        if len(df) == 0:
            return df
        if not isinstance(df.index, pd.MultiIndex):
            self.debug("converting to segmented index, this might take a while...")
            df.index = audformat.utils.to_segmented_index(df.index, allow_nat=False)
        return df

    def copy_flags(self, df_source, df_target):
        for flag in (
            "is_labeled",
            "is_test",
            "is_train",
            "is_val",
            "got_gender",
            "got_age",
            "got_speaker",
        ):
            if hasattr(df_source, flag):
                setattr(df_target, flag, getattr(df_source, flag))

    def get_labels(self):
        return ast.literal_eval(self.config["DATA"]["labels"])

    def continuous_to_categorical(self, series):
        """Discretize a continuous variable.

        Uses the labels and bins from the ini if present.

        :param series: a pandas series
        :return a pandas series with discretized values as categories
        """
        try:
            bins = ast.literal_eval(self.config["DATA"]["bins"])
            labels = ast.literal_eval(self.config["DATA"]["labels"])
        except KeyError:
            b1 = np.quantile(series, 0.33)
            b2 = np.quantile(series, 0.66)
            bins = [-1000000, b1, b2, 1000000]
            labels = ["0_low", "1_middle", "2_high"]
        result = np.digitize(series, bins) - 1
        result = pd.Series(result)
        for i, lab in enumerate(labels):
            result = result.replace(i, str(lab))
        return result.astype("category")

    def _bin_distributions(self, truths, preds):
        try:
            bins = ast.literal_eval(self.config["DATA"]["bins"])
        except KeyError:
            b1 = np.quantile(truths, 0.33)
            b2 = np.quantile(truths, 0.66)
            bins = [-1000000, b1, b2, 1000000]
        truths = np.digitize(truths, bins) - 1
        preds = np.digitize(preds, bins) - 1
        return truths, preds

    def df_to_cont_dict(self, df, column1, column2):
        """Convert a DataFrame with two continuous columns into a dictionary.

        Args:
            df (pd.DataFrame): Input DataFrame
            column1 (str): Name of the first continuous column
            column2 (str): Name of the second continuous column

        Returns:
            dict: {column1: [...], column2: [...]}

        Example:
        --------
        >>> util = Util()
        >>> df = pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]})
        >>> util.df_to_cont_dict(df, 'x', 'y')
        {'x': [1.0, 2.0], 'y': [3.0, 4.0]}
        """
        if column1 not in df.columns:
            self.error(f"Column '{column1}' not found in DataFrame")
        if column2 not in df.columns:
            self.error(f"Column '{column2}' not found in DataFrame")
        return {column1: df[column1].tolist(), column2: df[column2].tolist()}

    def df_to_categorical_dict(self, df, categorical_column, value_column):
        """Convert a DataFrame with a categorical and real-valued column to a dict.

        Args:
            df (pd.DataFrame): Input DataFrame
            categorical_column (str): Name of the categorical column (used as keys)
            value_column (str): Name of the real-valued column (used as values)

        Returns:
            tuple: (dict, float) where dict has categories as keys and value lists,
                   and float is the mean number of values per category

        Examples:
        --------
        >>> util = Util()
        >>> df = pd.DataFrame({
        ...     'emotion': ['happy', 'sad', 'happy', 'angry', 'sad'],
        ...     'intensity': [0.8, 0.6, 0.9, 0.7, 0.5]
        ... })
        >>> result_dict, mean_count = util.df_to_categorical_dict(
        ...     df, 'emotion', 'intensity')
        """
        if categorical_column not in df.columns:
            self.error(f"Column '{categorical_column}' not found in DataFrame")
        if value_column not in df.columns:
            self.error(f"Column '{value_column}' not found in DataFrame")

        result = {}
        for category in df[categorical_column].unique():
            mask = df[categorical_column] == category
            result[category] = df.loc[mask, value_column].tolist()

        mean_values_per_category = (
            sum(len(v) for v in result.values()) / len(result) if result else 0.0
        )
        return result, mean_values_per_category

    def is_dict_with_string_values(self, test_dict):
        try:
            return isinstance(test_dict, dict) and all(
                isinstance(v, str) for v in test_dict.values()
            )
        except (ValueError, SyntaxError):
            return False

    def map_labels(self, df, target, mapping):
        # mapping should be a dictionary; keys may encode comma-separated lists.
        keys = list(mapping.keys())
        for key in keys:
            if "," in key:
                key_list = [k.strip() for k in key.split(",")]
                for k in key_list:
                    mapping[k] = mapping[key]
                del mapping[key]
        df[target] = df[target].astype("string")
        df[target] = df[target].map(mapping)
        return df

    def scale_to_range(self, values, new_min=0, new_max=1):
        """Scale a list of numbers to a new min and max value.

        Args:
            values: List or array of numeric values to scale
            new_min: Target minimum value (default: 0)
            new_max: Target maximum value (default: 1)

        Returns:
            numpy.ndarray: Scaled values in the range [new_min, new_max]

        Examples:
        --------
        >>> util = Util()
        >>> values = [1, 2, 3, 4, 5]
        >>> scaled = util.scale_to_range(values, 0, 10)
        >>> # Returns: [0.0, 2.5, 5.0, 7.5, 10.0]
        """
        values = np.array(values, dtype=float)
        if len(values) == 0:
            return values
        old_min, old_max = np.min(values), np.max(values)
        if old_min == old_max:
            return np.full_like(values, (new_min + new_max) / 2)
        return (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def segment_silence(
    df: pd.DataFrame, with_borders: bool = True, remove_speaker_id: bool = False
) -> pd.DataFrame:
    """Take an already segmented (based on VAD) DataFrame and return the silence segments.

    Finds the gaps between speech segments within each file. Optionally includes
    the initial silence before the first speech segment (from t=0).

    Args:
        df: A DataFrame with a segmented audformat index (file, start, end).
        with_borders: If True, include the region from t=0 to the first speech
            segment start as an additional silence segment per file.
        remove_speaker_id: If True, set the speaker column to "silence" for all silence segments.

    Returns:
        A new DataFrame with one row per silence gap, All data columns are copied from the first segment entry per file.
    """
    if len(df) == 0:
        return df.iloc[0:0]

    silence_entries = []
    silence_data = []

    for file, file_df in df.groupby(level="file"):
        file_df = file_df.sort_index(level="start")
        starts = file_df.index.get_level_values("start")
        ends = file_df.index.get_level_values("end")
        first_row = file_df.iloc[0].to_dict()
        last_row = file_df.iloc[-1].to_dict()

        if with_borders and starts[0] > pd.Timedelta(0):
            silence_entries.append((file, pd.Timedelta(0), starts[0]))
            silence_data.append(first_row)

        for i in range(len(starts) - 1):
            gap_start = ends[i]
            gap_end = starts[i + 1]
            if gap_end > gap_start:
                silence_entries.append((file, gap_start, gap_end))
                silence_data.append(first_row)

        if with_borders and ends[-1] < file_df.index.get_level_values("end").max():
            silence_entries.append(
                (file, ends[-1], file_df.index.get_level_values("end").max())
            )
            silence_data.append(last_row)

    if not silence_entries:
        return df.iloc[0:0]

    new_index = pd.MultiIndex.from_tuples(
        silence_entries, names=["file", "start", "end"]
    )
    res_df = pd.DataFrame(silence_data, index=new_index)
    if remove_speaker_id and "speaker" in df.columns:
        res_df["speaker"] = "silence"
    return res_df
