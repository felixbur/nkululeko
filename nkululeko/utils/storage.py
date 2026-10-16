# storage.py - mixin for pickle, JSON, and DataFrame store I/O
import json
import os
import pickle

import audformat
import pandas as pd

from nkululeko.utils.pickle_integrity import save_checksum, verify_checksum


class StorageMixin:
    """Mixin providing file storage and I/O methods for Util."""

    def exist_pickle(self, name):
        store = self.get_path("store")
        name = "/".join([store, name]) + ".pkl"
        return os.path.isfile(name)

    def to_pickle(self, anyobject, name):
        store = self.get_path("store")
        name = "/".join([store, name]) + ".pkl"
        self.debug(f"saving {name}")
        with open(name, "wb") as handle:
            pickle.dump(anyobject, handle)
        save_checksum(name)

    def from_pickle(self, name):
        store = self.get_path("store")
        name = "/".join([store, name]) + ".pkl"
        self.debug(f"loading {name}")
        verify_checksum(name)
        with open(name, "rb") as handle:
            return pickle.load(handle)

    def write_store(self, df, storage, format):
        if format == "pkl":
            df.to_pickle(storage)
        elif format == "csv":
            from nkululeko.utils.sequences import is_sequence, write_sequence_csv

            if is_sequence(df):
                write_sequence_csv(df, storage)
            else:
                df.to_csv(storage)
        else:
            self.error(f"unknown store format: {format}")

    def get_store(self, name, format):
        if format == "pkl":
            return pd.read_pickle(name)
        elif format == "csv":
            df = pd.read_csv(name, nrows=0)
            if "frame" in df.columns:
                # frame sequences: audformat only rebuilds file/start/end
                from nkululeko.utils.sequences import read_sequence_csv

                return read_sequence_csv(name)
            return audformat.utils.read_csv(name)
        else:
            self.error(f"unknown store format: {format}")

    def save_to_store(self, df, name):
        store = self.get_path("store")
        store_format = self.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{name}.{store_format}"
        self.write_store(df, storage, store_format)

    def save_json(self, file: str, var: dict):
        """Save variable to json file.

        Args:
            file: path to json file
            var: dictionary to store
        """
        with open(file, "w", encoding="utf-8") as fp:
            json.dump(var, fp, ensure_ascii=False, indent=2)

    def read_json(self, file: str) -> object:
        """Read variable from json file.

        Args:
            file: path to json file

        Returns:
            content of json file
        """
        with open(file, "r") as fp:
            return json.load(fp)

    def read_first_line_floats(
        self, file_path: str, delimiter: str = None, strip_chars: str = None
    ) -> list:
        """Read the first line of a file and interpret it as a list of floats.

        Args:
            file_path: path to the file to read
            delimiter: delimiter to split the line (auto-detect if None)
            strip_chars: characters to strip from the line (default: whitespace)

        Returns:
            list: list of floats parsed from the first line

        Raises:
            FileNotFoundError: if the file does not exist
            ValueError: if the line cannot be parsed as floats
            IOError: if there are issues reading the file

        Examples:
        --------
        >>> util = Util()
        >>> floats = util.read_first_line_floats('data.txt')
        >>> floats = util.read_first_line_floats('data.csv', delimiter=',')
        """
        try:
            with open(file_path, "r") as fp:
                first_line = fp.readline()

                if not first_line:
                    self.debug(f"File {file_path} is empty")
                    return []

                first_line = (
                    first_line.strip(strip_chars)
                    if strip_chars is not None
                    else first_line.strip()
                )

                if not first_line:
                    self.debug(f"First line of {file_path} is empty after stripping")
                    return []

                # Auto-detect delimiter if not specified
                if delimiter is None:
                    for test_delimiter in [" ", ",", "\t", ";", "|"]:
                        if test_delimiter in first_line:
                            delimiter = test_delimiter
                            break

                string_values = (
                    first_line.split(delimiter)
                    if delimiter is not None
                    else [first_line]
                )

                float_values = []
                for value in string_values:
                    value = value.strip()
                    if value:
                        try:
                            float_values.append(float(value))
                        except ValueError as e:
                            self.error(
                                f"Cannot convert '{value}' to float in file {file_path}: {e}"
                            )

                self.debug(f"Read {len(float_values)} floats from {file_path}")
                return float_values

        except FileNotFoundError:
            self.error(f"File not found: {file_path}")
        except IOError as e:
            self.error(f"Error reading file {file_path}: {e}")
        except Exception as e:
            self.error(f"Unexpected error reading floats from {file_path}: {e}")
