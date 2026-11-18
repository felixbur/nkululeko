"""Tests for nkululeko/data/dataset_csv.py -- a missing 'file' value.

A CSV-backed database whose 'file' column has an empty/missing value
(e.g. produced by a process_database.py script before the real audio
is available) used to crash deep inside os.path.join() with a cryptic
`TypeError: ... not 'NAType'`, once DATA.<name>.audio_path was joined
onto the index further down in load(). Dataset_CSV now checks for this
right after the index is built and raises a clear error instead.
"""

import pandas as pd
import pytest

from nkululeko.data.dataset_csv import Dataset_CSV
from nkululeko.utils.errors import NkululukoError


def _make_dataset(config_values):
    ds = Dataset_CSV.__new__(Dataset_CSV)
    ds.name = "mydb"

    class _FakeUtil:
        def config_val_data(self, name, key, default):
            return config_values.get(key, default)

        def error(self, message):
            raise NkululukoError(f"ERROR: test: {message}")

        def debug(self, message):
            pass

    ds.util = _FakeUtil()
    return ds


class TestCheckFileIndex:
    def test_no_missing_values_passes_silently(self):
        ds = _make_dataset({})
        df = pd.DataFrame({"label": ["a", "b"]}, index=["/a.wav", "/b.wav"])
        df.index.name = "file"
        ds._check_file_index(df)  # must not raise

    def test_missing_value_raises_with_count_and_name(self):
        ds = _make_dataset({})
        df = pd.DataFrame({"label": ["a", "b", "c"]}, index=[pd.NA, "/b.wav", pd.NA])
        df.index.name = "file"
        with pytest.raises(NkululukoError, match="mydb") as exc_info:
            ds._check_file_index(df)
        assert "2" in str(exc_info.value)
        assert "file" in str(exc_info.value)

    def test_missing_value_in_segmented_index_raises(self):
        import audformat

        ds = _make_dataset({})
        index = audformat.segmented_index(
            files=[pd.NA, "/b.wav"], starts=[0.0, 0.0], ends=[1.0, 1.0]
        )
        df = pd.DataFrame({"label": ["a", "b"]}, index=index)
        with pytest.raises(NkululukoError, match="mydb"):
            ds._check_file_index(df)


class TestLoadRaisesOnMissingFile:
    def test_load_raises_a_clear_error_not_a_typeerror(self, tmp_path):
        """GH follow-up from the EMOVOME example (#492): a
        process_database.py-style CSV with an empty 'file' column (no
        audio available yet) must fail with a clear message, not a
        TypeError from inside os.path.join()."""
        csv_path = tmp_path / "mydb.csv"
        pd.DataFrame({"file": ["", ""], "label": ["a", "b"]}).to_csv(
            csv_path, index=False
        )

        ds = _make_dataset(
            {
                "": str(csv_path),
                "audio_path": "./",
                "absolute_path": "True",
            }
        )

        with pytest.raises(NkululukoError, match="mydb") as exc_info:
            ds.load()
        assert "2" in str(exc_info.value)
