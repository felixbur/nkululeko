# test_dataframe.py - unit tests for nkululeko/utils/dataframe.py
import configparser
import math
import unittest
from datetime import timedelta

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
        self.assertTrue(all(math.isclose(v, 0.5) for v in result))

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
            bins_labels='bins = [-1000000, 0.5, 1000000]\nlabels = ["low", "high"]'
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
        df = pd.DataFrame(
            {
                "emotion": ["happy", "sad", "happy"],
                "score": [0.8, 0.6, 0.9],
            }
        )
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

    # --- Additional tests for uncovered lines ---

    def test_is_numeric(self):
        u = make_util()
        s = pd.Series([1.0, 2.0, 3.0])
        self.assertTrue(u.is_numeric(s))

        s_str = pd.Series(["a", "b", "c"])
        self.assertFalse(u.is_numeric(s_str))

    def test_make_segmented_index_empty(self):
        u = make_util()
        df = pd.DataFrame()
        result = u.make_segmented_index(df)
        self.assertTrue(result.empty)

    def test_get_labels(self):
        c = configparser.ConfigParser()
        c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
labels = ["happy", "sad"]
[MODEL]
type = svm
[FEATS]
type = os
""")
        glob_conf.config = c
        u = Util("test")
        result = u.get_labels()
        self.assertEqual(result, ["happy", "sad"])

    def test_df_to_cont_dict_missing_column1(self):
        u = make_util()
        df = pd.DataFrame({"y": [3.0, 4.0]})
        with self.assertRaises(SystemExit):
            u.df_to_cont_dict(df, "x", "y")

    def test_df_to_cont_dict_missing_column2(self):
        u = make_util()
        df = pd.DataFrame({"x": [1.0, 2.0]})
        with self.assertRaises(SystemExit):
            u.df_to_cont_dict(df, "x", "y")

    def test_df_to_categorical_dict_missing_categorical_column(self):
        u = make_util()
        df = pd.DataFrame({"score": [0.8, 0.6]})
        with self.assertRaises(SystemExit):
            u.df_to_categorical_dict(df, "emotion", "score")

    def test_df_to_categorical_dict_missing_value_column(self):
        u = make_util()
        df = pd.DataFrame({"emotion": ["happy", "sad"]})
        with self.assertRaises(SystemExit):
            u.df_to_categorical_dict(df, "emotion", "score")

    def test_is_dict_with_string_values_exception(self):
        u = make_util()
        # Create a dict that might trigger exception in all() check
        # This tests the exception handling
        result = u.is_dict_with_string_values({"a": object()})
        self.assertFalse(result)


class TestSegmentSilence(unittest.TestCase):
    """Unit tests for segment_silence() in nkululeko/utils/dataframe.py."""

    @classmethod
    def setUpClass(cls):
        from nkululeko.utils.dataframe import segment_silence

        cls.segment_silence = staticmethod(segment_silence)

    def _make_seg_df(self, entries, columns=None):
        """Build a segmented-index DataFrame from (file, start_sec, end_sec) tuples."""
        idx = pd.MultiIndex.from_tuples(
            [(f, timedelta(seconds=s), timedelta(seconds=e)) for f, s, e in entries],
            names=["file", "start", "end"],
        )
        data = {col: [None] * len(entries) for col in (columns or [])}
        return pd.DataFrame(data, index=idx)

    def test_segment_silence_basic_gap(self):
        """Two speech segments separated by a gap yield exactly one silence row."""
        df = self._make_seg_df([("f.wav", 1.0, 3.0), ("f.wav", 5.0, 7.0)])
        result = self.segment_silence(df, with_borders=False)
        self.assertEqual(len(result), 1)
        self.assertEqual(
            result.index[0], ("f.wav", timedelta(seconds=3), timedelta(seconds=5))
        )

    def test_segment_silence_multiple_files(self):
        """Each file's gaps are computed independently."""
        df = self._make_seg_df(
            [
                ("a.wav", 0.0, 2.0),
                ("a.wav", 4.0, 6.0),
                ("b.wav", 1.0, 3.0),
                ("b.wav", 7.0, 9.0),
            ]
        )
        result = self.segment_silence(df, with_borders=False)
        self.assertEqual(len(result), 2)
        files = result.index.get_level_values("file").tolist()
        self.assertIn("a.wav", files)
        self.assertIn("b.wav", files)
        # a.wav gap: 2→4, b.wav gap: 3→7
        a_row = result[result.index.get_level_values("file") == "a.wav"]
        self.assertEqual(a_row.index[0][1], timedelta(seconds=2))
        self.assertEqual(a_row.index[0][2], timedelta(seconds=4))
        b_row = result[result.index.get_level_values("file") == "b.wav"]
        self.assertEqual(b_row.index[0][1], timedelta(seconds=3))
        self.assertEqual(b_row.index[0][2], timedelta(seconds=7))

    def test_segment_silence_with_borders_leading(self):
        """with_borders=True adds a leading silence when first segment starts after t=0."""
        df = self._make_seg_df([("f.wav", 2.0, 4.0), ("f.wav", 6.0, 8.0)])
        result = self.segment_silence(df, with_borders=True)
        # Should have: leading silence (0→2) + gap (4→6)
        self.assertEqual(len(result), 2)
        starts = result.index.get_level_values("start").tolist()
        self.assertIn(timedelta(0), starts)

    def test_segment_silence_with_borders_no_leading(self):
        """with_borders=True does NOT add a leading silence when first segment starts at t=0."""
        df = self._make_seg_df([("f.wav", 0.0, 2.0), ("f.wav", 4.0, 6.0)])
        result = self.segment_silence(df, with_borders=True)
        # Only the gap (2→4), no leading silence
        self.assertEqual(len(result), 1)
        self.assertNotIn(timedelta(0), result.index.get_level_values("start").tolist())

    def test_segment_silence_adjacent_segments(self):
        """Adjacent segments (end == next start) produce no silence row."""
        df = self._make_seg_df([("f.wav", 0.0, 3.0), ("f.wav", 3.0, 6.0)])
        result = self.segment_silence(df, with_borders=False)
        self.assertEqual(len(result), 0)

    def test_segment_silence_overlapping_segments(self):
        """Overlapping segments produce no silence row."""
        # Second interval starts before the first one ends (true overlap).
        df = self._make_seg_df([("f.wav", 0.0, 5.0), ("f.wav", 3.0, 7.0)])
        result = self.segment_silence(df, with_borders=False)
        self.assertEqual(len(result), 0)

    def test_segment_silence_empty(self):
        """An empty input DataFrame returns an empty DataFrame."""
        df = self._make_seg_df([])
        result = self.segment_silence(df, with_borders=False)
        self.assertEqual(len(result), 0)

    def test_segment_silence_empty_with_borders(self):
        """Empty input with borders enabled still returns an empty DataFrame."""
        df = self._make_seg_df([])
        result = self.segment_silence(df, with_borders=True)
        self.assertEqual(len(result), 0)

    def test_segment_silence_single_segment(self):
        """A single segment per file with no gaps yields no silence rows."""
        df = self._make_seg_df([("f.wav", 1.0, 3.0)])
        result = self.segment_silence(df, with_borders=False)
        self.assertEqual(len(result), 0)

    def test_segment_silence_remove_speaker_id(self):
        """remove_speaker_id=True sets the speaker column to 'silence'."""
        df = self._make_seg_df(
            [("f.wav", 0.0, 2.0), ("f.wav", 4.0, 6.0)],
            columns=["speaker"],
        )
        df["speaker"] = "spk1"
        result = self.segment_silence(df, with_borders=False, remove_speaker_id=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["speaker"].iloc[0], "silence")


if __name__ == "__main__":
    unittest.main()
