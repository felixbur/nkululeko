# test_dataframe.py - unit tests for nkululeko/utils/dataframe.py
import configparser
import unittest

import numpy as np
import pandas as pd

import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


def make_util(bins_labels=""):
    c = configparser.ConfigParser()
    c.read_string(
        f"""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
{bins_labels}
[MODEL]
type = svm
[FEATS]
type = os
"""
    )
    glob_conf.config = c
    return Util("test")


class TestDataFrameMixin(unittest.TestCase):

    # --- is_categorical ---

    def test_is_categorical_object(self):
        u = make_util()
        s = pd.Series(["a", "b", "c"])
        self.assertTrue(u.is_categorical(s))

    def test_is_categorical_string_dtype(self):
        u = make_util()
        s = pd.Series(["a", "b"], dtype="string")
        self.assertTrue(u.is_categorical(s))

    def test_is_categorical_category(self):
        u = make_util()
        s = pd.Series(["a", "b"], dtype="category")
        self.assertTrue(u.is_categorical(s))

    def test_is_categorical_numeric_is_false(self):
        u = make_util()
        s = pd.Series([1.0, 2.0, 3.0])
        self.assertFalse(u.is_categorical(s))

    # --- scale_to_range ---

    def test_scale_to_range_basic(self):
        u = make_util()
        result = u.scale_to_range([1, 2, 3, 4, 5], 0, 1)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[-1], 1.0)

    def test_scale_to_range_custom_bounds(self):
        u = make_util()
        result = u.scale_to_range([0, 10], -1, 1)
        self.assertAlmostEqual(result[0], -1.0)
        self.assertAlmostEqual(result[-1], 1.0)

    def test_scale_to_range_constant_values(self):
        u = make_util()
        result = u.scale_to_range([5, 5, 5], 0, 1)
        # all values become midpoint
        self.assertTrue(all(v == 0.5 for v in result))

    def test_scale_to_range_empty(self):
        u = make_util()
        result = u.scale_to_range([])
        self.assertEqual(len(result), 0)

    # --- continuous_to_categorical ---

    def test_continuous_to_categorical_default_bins(self):
        u = make_util()
        series = pd.Series(np.linspace(0, 1, 100))
        result = u.continuous_to_categorical(series)
        self.assertEqual(str(result.dtype), "category")
        unique_vals = set(result.unique())
        self.assertEqual(unique_vals, {"0_low", "1_middle", "2_high"})

    def test_continuous_to_categorical_custom_bins(self):
        u = make_util(
            bins_labels="bins = [-1000000, 0.5, 1000000]\nlabels = [\"low\", \"high\"]"
        )
        series = pd.Series([0.0, 0.3, 0.7, 1.0])
        result = u.continuous_to_categorical(series)
        self.assertIn("low", result.values)
        self.assertIn("high", result.values)

    # --- _bin_distributions ---

    def test_bin_distributions_default(self):
        u = make_util()
        truths = np.linspace(0, 1, 100)
        preds = np.linspace(0, 1, 100)
        t_binned, p_binned = u._bin_distributions(truths, preds)
        # 3 bins → values 0, 1, 2
        self.assertEqual(set(np.unique(t_binned)), {0, 1, 2})

    def test_bin_distributions_custom(self):
        u = make_util(bins_labels="bins = [-1000000, 0.5, 1000000]")
        truths = np.array([0.1, 0.9])
        preds = np.array([0.2, 0.8])
        t_binned, p_binned = u._bin_distributions(truths, preds)
        self.assertEqual(t_binned[0], 0)
        self.assertEqual(t_binned[1], 1)

    # --- copy_flags ---

    def test_copy_flags(self):
        u = make_util()
        src = pd.DataFrame()
        src.is_train = True
        src.is_test = False
        dst = pd.DataFrame()
        u.copy_flags(src, dst)
        self.assertTrue(dst.is_train)
        self.assertFalse(dst.is_test)

    def test_copy_flags_missing_attr_ignored(self):
        u = make_util()
        src = pd.DataFrame()  # no flags set
        dst = pd.DataFrame()
        u.copy_flags(src, dst)  # should not raise
        self.assertFalse(hasattr(dst, "is_train"))

    # --- df_to_cont_dict ---

    def test_df_to_cont_dict(self):
        u = make_util()
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        result = u.df_to_cont_dict(df, "x", "y")
        self.assertEqual(result["x"], [1.0, 2.0])
        self.assertEqual(result["y"], [3.0, 4.0])

    # --- df_to_categorical_dict ---

    def test_df_to_categorical_dict(self):
        u = make_util()
        df = pd.DataFrame({
            "emotion": ["happy", "sad", "happy"],
            "score": [0.8, 0.6, 0.9],
        })
        result, mean_count = u.df_to_categorical_dict(df, "emotion", "score")
        self.assertIn("happy", result)
        self.assertEqual(len(result["happy"]), 2)
        self.assertAlmostEqual(mean_count, 1.5)

    # --- is_dict_with_string_values ---

    def test_is_dict_with_string_values_true(self):
        u = make_util()
        self.assertTrue(u.is_dict_with_string_values({"a": "x", "b": "y"}))

    def test_is_dict_with_string_values_false(self):
        u = make_util()
        self.assertFalse(u.is_dict_with_string_values({"a": 1, "b": "y"}))

    def test_is_dict_with_string_values_not_dict(self):
        u = make_util()
        self.assertFalse(u.is_dict_with_string_values("not a dict"))

    # --- map_labels ---

    def test_map_labels_basic(self):
        u = make_util()
        df = pd.DataFrame({"emotion": ["happy", "sad", "angry"]})
        mapping = {"happy": "pos", "sad": "neg", "angry": "neg"}
        result = u.map_labels(df, "emotion", mapping)
        self.assertEqual(list(result["emotion"]), ["pos", "neg", "neg"])

    def test_map_labels_comma_key(self):
        u = make_util()
        df = pd.DataFrame({"emotion": ["happy", "excited"]})
        mapping = {"happy, excited": "pos"}
        result = u.map_labels(df, "emotion", mapping)
        self.assertEqual(list(result["emotion"]), ["pos", "pos"])


if __name__ == "__main__":
    unittest.main()
