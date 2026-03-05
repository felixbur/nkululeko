# dataframe.py - mixin for DataFrame, label, and numeric helpers
import ast

import audformat
import numpy as np
import pandas as pd


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
        """Check if a dataframe column is numeric."""
        return (
            pd_series.dtype.name == "int64"
            or pd_series.dtype.name == "float64"
            or isinstance(pd_series.dtype, pd.Int64Dtype)
            or isinstance(pd_series.dtype, pd.Float64Dtype)
        )
     
    def make_segmented_index(self, df):
        if len(df) == 0:
            return df
        if not isinstance(df.index, pd.MultiIndex):
            self.debug("converting to segmented index, this might take a while...")
            df.index = audformat.utils.to_segmented_index(df.index, allow_nat=False)
        return df

    def copy_flags(self, df_source, df_target):
        for flag in ("is_labeled", "is_test", "is_train", "is_val", "got_gender", "got_age", "got_speaker"):
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
