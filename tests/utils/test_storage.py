# test_storage.py - unit tests for nkululeko/utils/storage.py
import configparser
import json
import os
import pickle
import tempfile
import unittest

import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


def make_util():
    c = configparser.ConfigParser()
    c.read_string("""
[EXP]
name = test
root = /tmp
[DATA]
databases = ["emodb"]
target = emotion
[MODEL]
type = svm
[FEATS]
type = os
""")
    glob_conf.config = c
    return Util("test")


class TestStorageMixin(unittest.TestCase):

    # --- save_json / read_json round-trip ---

    def test_save_and_read_json(self):
        u = make_util()
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            u.save_json(path, data)
            result = u.read_json(path)
            self.assertEqual(result, data)
        finally:
            os.unlink(path)

    def test_save_json_unicode(self):
        u = make_util()
        data = {"greeting": "héllo wörld"}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            u.save_json(path, data)
            result = u.read_json(path)
            self.assertEqual(result["greeting"], "héllo wörld")
        finally:
            os.unlink(path)

    # --- read_first_line_floats ---

    def _write_tmp(self, content):
        f = tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def test_read_first_line_floats_space(self):
        u = make_util()
        path = self._write_tmp("0.1 0.2 0.3\n")
        try:
            result = u.read_first_line_floats(path)
            self.assertEqual(result, [0.1, 0.2, 0.3])
        finally:
            os.unlink(path)

    def test_read_first_line_floats_comma(self):
        u = make_util()
        path = self._write_tmp("1.0,2.0,3.0\n")
        try:
            result = u.read_first_line_floats(path, delimiter=",")
            self.assertEqual(result, [1.0, 2.0, 3.0])
        finally:
            os.unlink(path)

    def test_read_first_line_floats_auto_detect_comma(self):
        u = make_util()
        path = self._write_tmp("4.0,5.0,6.0\n")
        try:
            result = u.read_first_line_floats(path)
            self.assertEqual(result, [4.0, 5.0, 6.0])
        finally:
            os.unlink(path)

    def test_read_first_line_floats_single_value(self):
        u = make_util()
        path = self._write_tmp("3.14\n")
        try:
            result = u.read_first_line_floats(path)
            self.assertEqual(result, [3.14])
        finally:
            os.unlink(path)

    def test_read_first_line_floats_empty_file(self):
        u = make_util()
        path = self._write_tmp("")
        try:
            result = u.read_first_line_floats(path)
            self.assertEqual(result, [])
        finally:
            os.unlink(path)

    def test_read_first_line_floats_only_reads_first_line(self):
        u = make_util()
        path = self._write_tmp("1.0 2.0\n3.0 4.0\n")
        try:
            result = u.read_first_line_floats(path)
            self.assertEqual(result, [1.0, 2.0])
        finally:
            os.unlink(path)

    # --- write_store / get_store round-trip (pkl) ---

    def test_write_and_get_store_pkl(self):
        import pandas as pd

        u = make_util()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            u.write_store(df, path, "pkl")
            result = u.get_store(path, "pkl")
            self.assertEqual(list(result["a"]), [1, 2])
        finally:
            os.unlink(path)

    # --- exist_pickle / to_pickle / from_pickle ---

    def test_pickle_round_trip(self):
        u = make_util()
        obj = {"hello": "world", "nums": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as tmpdir:
            # point store path to tmpdir
            u.config["EXP"]["root"] = tmpdir
            u.config["EXP"]["name"] = "pickletest"
            name = "myobj"
            u.to_pickle(obj, name)
            self.assertTrue(u.exist_pickle(name))
            loaded = u.from_pickle(name)
            self.assertEqual(loaded, obj)

    def test_exist_pickle_false_when_missing(self):
        u = make_util()
        with tempfile.TemporaryDirectory() as tmpdir:
            u.config["EXP"]["root"] = tmpdir
            u.config["EXP"]["name"] = "pickletest"
            self.assertFalse(u.exist_pickle("nonexistent"))


if __name__ == "__main__":
    unittest.main()
