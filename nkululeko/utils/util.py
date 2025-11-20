# util.py
import ast
import configparser
import json
import logging
import os.path
import pickle
import sys

import numpy as np
import pandas as pd

import audeer
import audformat


class Util:
    # a list of words that need not to be warned upon if default values are
    # used
    stopvals = [
        "all",
        False,
        "False",
        True,
        "True",
        "classification",
        "png",
        "audio_path",
        "kde",
        "pkl",
        "eGeMAPSv02",
        "functionals",
        "n_jobs",
    ]

    def __init__(self, caller=None, has_config=True):
        self.logger = None
        if caller is not None:
            self.caller = caller
        else:
            self.caller = ""
        self.config = None
        if has_config:
            try:
                import nkululeko.glob_conf as glob_conf

                self.config = glob_conf.config
                self.got_data_roots = self.config_val("DATA", "root_folders", False)
                if self.got_data_roots:
                    # if there is a global data rootfolder file, read from
                    # there
                    if not os.path.isfile(self.got_data_roots):
                        self.error(f"no such file: {self.got_data_roots}")
                    self.data_roots = configparser.ConfigParser()
                    self.data_roots.read(self.got_data_roots)
            except ModuleNotFoundError as e:
                self.error(e)
                self.config = None
                self.got_data_roots = False
            except AttributeError as e:
                self.error(e)
                self.config = None
                self.got_data_roots = False

        self.setup_logging()
        # self.logged_configs = set()

    def setup_logging(self):
        # Setup logging
        logger = logging.getLogger(__name__)
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)  # Set the desired logging level

            # Create a console handler
            console_handler = logging.StreamHandler()

            # Create a simple formatter that only shows the message
            class SimpleFormatter(logging.Formatter):
                def format(self, record):
                    return record.getMessage()

            # Set the formatter for the console handler
            console_handler.setFormatter(SimpleFormatter())

            # Add the console handler to the logger
            logger.addHandler(console_handler)
        self.logger = logger

    def get_path(self, entry):
        """This method allows the user to get the directory path for the given argument."""
        if self.config is None:
            # If no configuration file is provided, use default paths
            if entry == "fig_dir":
                dir_name = "./images/"
            elif entry == "res_dir":
                dir_name = "./results/"
            elif entry == "model_dir":
                dir_name = "./models/"
            elif entry == "cache":
                dir_name = "./cache/"
            else:
                dir_name = "./store/"
        else:
            root = os.path.join(self.config["EXP"]["root"], "")
            name = self.config["EXP"]["name"]
            try:
                entryn = self.config["EXP"][entry]
            except KeyError:
                # some default values
                if entry == "fig_dir":
                    entryn = "images/"
                elif entry == "res_dir":
                    entryn = "results/"
                elif entry == "model_dir":
                    entryn = "models/"
                elif entry == "cache":
                    entryn = "cache/"
                else:
                    entryn = "store/"

            # Expand image, model and result directories with run index
            if entry == "fig_dir" or entry == "res_dir" or entry == "model_dir":
                run = self.config_val("EXP", "run", 0)
                entryn = entryn + f"run_{run}/"

            dir_name = f"{root}{name}/{entryn}"

        audeer.mkdir(dir_name)
        return dir_name

    def config_val_data(self, dataset, key, default):
        """Retrieve a configuration value for datasets.

        If the value is present in the experiment configuration it will be used, else
        we look in a global file specified by the root_folders value.
        """
        import nkululeko.glob_conf as glob_conf

        configuration = glob_conf.config
        try:
            if len(key) > 0:
                return configuration["DATA"][dataset + "." + key].strip("'\"")
            else:
                return configuration["DATA"][dataset].strip("'\"")
        except KeyError:
            if self.got_data_roots:
                try:
                    if len(key) > 0:
                        return self.data_roots["DATA"][dataset + "." + key].strip("'\"")
                    else:
                        return self.data_roots["DATA"][dataset].strip("'\"")
                except KeyError:
                    if default not in self.stopvals:
                        self.debug(
                            f"value for {key} not found, using default:" f" {default}"
                        )
                    return default
            if default not in self.stopvals:
                self.debug(f"value for {key} not found, using default: {default}")
            return default

    def set_config(self, config):
        self.config = config
        # self.logged_configs.clear()

    def get_save_name(self):
        """Return a relative path to a name to save the experiment."""
        store = self.get_path("store")
        return f"{store}/{self.get_exp_name()}.pkl"

    def get_pred_name(self):
        results_dir = self.get_path("res_dir")
        target = self.get_target_name()
        pred_name = self.get_model_description()
        return f"{results_dir}/pred_{target}_{pred_name}"

    def print_results_to_store(self, name: str, contents: str) -> str:
        """Write contents to a result file.

        Args:
            name (str): the (sub) name of the file_

        Returns:
            str: The path to the file
        """
        results_dir = self.get_path("res_dir")
        pred_name = self.get_model_description()
        path = os.path.join(results_dir, f"{name}_{pred_name}.txt")
        with open(path, "a") as f:
            f.write(contents)

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

    def get_name(self):
        """Get the name of the experiment."""
        return self.config["EXP"]["name"]

    def get_exp_dir(self):
        """Get the experiment directory."""
        root = os.path.join(self.config["EXP"]["root"], "")
        name = self.config["EXP"]["name"]
        dir_name = f"{root}/{name}"
        audeer.mkdir(dir_name)
        return dir_name

    def get_res_dir(self):
        home_dir = self.get_exp_dir()
        dir_name = f"{home_dir}/results/"
        audeer.mkdir(dir_name)
        return dir_name

    def make_segmented_index(self, df):
        if len(df) == 0:
            return df
        if not isinstance(df.index, pd.MultiIndex):
            self.debug("converting to segmented index, this might take a while...")
            df.index = audformat.utils.to_segmented_index(df.index, allow_nat=False)
        return df

    def _get_value_descript(self, section, name):
        if self.config_val(section, name, False):
            val = self.config_val(section, name, False)
            val = str(val).strip(".")
            return f"_{name}-{str(val)}"
        return ""

    def get_data_name(self):
        """Get a string as name from all databases that are used."""
        return "_".join(ast.literal_eval(self.config["DATA"]["databases"]))

    def get_feattype_name(self):
        """Get a string as name from all feature sets that are used."""
        return "_".join(ast.literal_eval(self.config["FEATS"]["type"]))

    def get_exp_name(self, only_train=False, only_data=False):
        trains_val = self.config_val("DATA", "trains", False)
        if only_train and trains_val:
            # try to get only the train tables
            ds = "-".join(ast.literal_eval(self.config["DATA"]["trains"]))
        else:
            ds = "-".join(ast.literal_eval(self.config["DATA"]["databases"]))
        return_string = f"{ds}"
        if not only_data:
            mt = self.get_model_description()
            target = self.get_target_name()
            return_string = return_string + "_" + target + "_" + mt
        return return_string.replace("__", "_")

    def get_target_name(self):
        """Get a string as name from all target sets that are used."""
        return self.config["DATA"]["target"]

    def get_model_type(self):
        try:
            return self.config["MODEL"]["type"]
        except KeyError:
            return ""

    def get_model_description(self):
        mt = ""
        try:
            mt = f'{self.config["MODEL"]["type"]}'
        except KeyError:
            # no model type given
            pass
        # ft = "_".join(ast.literal_eval(self.config["FEATS"]["type"]))
        ft_value = self.config["FEATS"]["type"]
        if (
            isinstance(ft_value, str)
            and ft_value.startswith("[")
            and ft_value.endswith("]")
        ):
            ft = "-".join(ast.literal_eval(ft_value))
        else:
            ft = ft_value
        ft += "_"
        layer_string = ""
        layer_s = self.config_val("MODEL", "layers", False)
        if layer_s:
            layers = ast.literal_eval(layer_s)
            if type(layers) is list:
                layers_list = layers.copy()
                # construct a dict with layer names
                keys = [str(i) for i in range(len(layers_list))]
                layers = dict(zip(keys, layers_list))
            sorted_layers = sorted(layers.items(), key=lambda x: x[1])
            for layer in sorted_layers:
                layer_string += f"{str(layer[1])}-"
        return_string = f"{mt}_{ft}{layer_string[:-1]}"
        options = [
            ["MODEL", "C_val"],
            ["MODEL", "kernel"],
            ["MODEL", "drop"],
            ["MODEL", "class_weight"],
            ["MODEL", "loss"],
            ["MODEL", "logo"],
            ["MODEL", "learning_rate"],
            ["MODEL", "k_fold_cross"],
            ["FEATS", "balancing"],
            ["FEATS", "scale"],
            ["FEATS", "set"],
            ["FEATS", "wav2vec2.layer"],
        ]
        for option in options:
            return_string += self._get_value_descript(option[0], option[1]).replace(
                ".", "-"
            )
            # prevent double underscores
            return_string = return_string.replace("__", "_")
            # remove trailing underscores in the end
            return_string = return_string.strip("_")

        return return_string

    def get_plot_name(self):
        try:
            plot_name = self.config["PLOT"]["name"]
        except KeyError:
            plot_name = self.get_exp_name()
        return plot_name

    def exp_is_classification(self):
        type = self.config_val("EXP", "type", "classification")
        if type == "classification":
            return True
        return False

    def error(self, message):
        if self.logger is not None:
            self.logger.error(f"ERROR: {self.caller}: {message}")
        else:
            print(f"ERROR: {message}")
        sys.exit()

    def warn(self, message):
        if self.logger is not None:
            self.logger.warning(f"WARNING: {self.caller}: {message}")
        else:
            print(f"WARNING: {message}", flush=True)

    def debug(self, message):
        if self.logger is not None:
            self.logger.debug(f"DEBUG: {self.caller}: {message}")
        else:
            print(f"DEBUG: {message}", flush=True)

    def set_config_val(self, section, key, value):
        try:
            # does the section already exists?
            self.config[section][key] = str(value)
        except KeyError:
            self.config.add_section(section)
            self.config[section][key] = str(value)

    def check_df(self, i, df):
        """Check a dataframe."""
        print(f"check {i}: {df.shape}")
        print(df.head(1))

    def config_val(self, section, key, default):
        if self.config is None:
            return default
        try:
            return self.config[section][key]
        except KeyError:
            if default not in self.stopvals:
                self.debug(f"value for {key} is not found, using default: {default}")
            return default

    @classmethod
    def reset_logged_configs(cls):
        cls.logged_configs.clear()

    def config_val_list(self, section, key, default):
        try:
            return ast.literal_eval(self.config[section][key])
        except KeyError:
            if default not in self.stopvals:
                self.debug(f"value for {key} not found, using default: {default}")
            return default

    def get_labels(self):
        return ast.literal_eval(self.config["DATA"]["labels"])

    def continuous_to_categorical(self, series):
        """Discretize a categorical variable.

        Uses the labels and bins from the ini if present

        :param series: a pandas series
        :return a pandas series with discretized values as categories
        """
        try:
            bins = ast.literal_eval(self.config["DATA"]["bins"])
            labels = ast.literal_eval(self.config["DATA"]["labels"])
        except KeyError:
            # if no binning is given, simply take three bins
            b1 = np.quantile(series, 0.33)
            b2 = np.quantile(series, 0.66)
            bins = [-1000000, b1, b2, 1000000]
            labels = ["0_low", "1_middle", "2_high"]
        result = np.digitize(series, bins) - 1
        result = pd.Series(result)
        for i, lab in enumerate(labels):
            result = result.replace(i, str(lab))
        result = result.astype("category")
        return result

    def _bin_distributions(self, truths, preds):
        try:
            bins = ast.literal_eval(self.config["DATA"]["bins"])
        except KeyError:
            # if no binning is given, simply take three bins, based on truth
            b1 = np.quantile(truths, 0.33)
            b2 = np.quantile(truths, 0.66)
            bins = [-1000000, b1, b2, 1000000]
        truths = np.digitize(truths, bins) - 1
        preds = np.digitize(preds, bins) - 1
        return truths, preds

    def print_best_results(self, best_reports):
        res_dir = self.get_res_dir()
        # go one level up above the "run" level
        all = ""
        vals = np.empty(0)
        for report in best_reports:
            all += str(report.result.test) + ", "
            vals = np.append(vals, report.result.test)
        file_name = f"{res_dir}{self.get_exp_name()}_runs.txt"
        output = (
            f"{all}"
            + f"\nmean: {vals.mean():.3f}, std: {vals.std():.3f}, "
            + f"max: {vals.max():.3f}, max_index: {vals.argmax()}"
        )
        with open(file_name, "w") as text_file:
            text_file.write(output)
        self.debug(output)

    def exist_pickle(self, name):
        store = self.get_path("store")
        name = "/".join([store, name]) + ".pkl"
        if os.path.isfile(name):
            return True
        return False

    def to_pickle(self, anyobject, name):
        store = self.get_path("store")
        name = "/".join([store, name]) + ".pkl"
        self.debug(f"saving {name}")
        with open(name, "wb") as handle:
            pickle.dump(anyobject, handle)

    def from_pickle(self, name):
        store = self.get_path("store")
        name = "/".join([store, name]) + ".pkl"
        self.debug(f"loading {name}")
        with open(name, "rb") as handle:
            any_opject = pickle.load(handle)
        return any_opject

    def write_store(self, df, storage, format):
        if format == "pkl":
            df.to_pickle(storage)
        elif format == "csv":
            df.to_csv(storage)
        else:
            self.error(f"unkown store format: {format}")

    def get_store(self, name, format):
        if format == "pkl":
            return pd.read_pickle(name)
        elif format == "csv":
            return audformat.utils.read_csv(name)
        else:
            self.error(f"unknown store format: {format}")

    def save_to_store(self, df, name):
        store = self.get_path("store")
        store_format = self.config_val("FEATS", "store_format", "pkl")
        storage = f"{store}{name}.{store_format}"
        self.write_store(df, storage, store_format)

    def copy_flags(self, df_source, df_target):
        if hasattr(df_source, "is_labeled"):
            df_target.is_labeled = df_source.is_labeled
        if hasattr(df_source, "is_test"):
            df_target.is_test = df_source.is_test
        if hasattr(df_source, "is_train"):
            df_target.is_train = df_source.is_train
        if hasattr(df_source, "is_val"):
            df_target.is_val = df_source.is_val
        if hasattr(df_source, "got_gender"):
            df_target.got_gender = df_source.got_gender
        if hasattr(df_source, "got_age"):
            df_target.got_age = df_source.got_age
        if hasattr(df_source, "got_speaker"):
            df_target.got_speaker = df_source.got_speaker

    def high_is_good(self):
        """check how to interpret results"""
        if self.exp_is_classification():
            return True
        else:
            measure = self.config_val("MODEL", "measure", "mse")
            if measure == "mse" or measure == "mae":
                return False
            elif measure == "ccc":
                return True
            else:
                self.error(f"unknown measure: {measure}")

    def to_3_digits(self, x):
        """Given a float, return this to 3 digits."""
        x = float(x)
        return (int(x * 1000)) / 1000.0

    def to_3_digits_str(self, x):
        """Given a float, return this to 3 digits as string without integer number."""
        return str(self.to_3_digits(x))[1:]

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
        >>> # Read space-separated floats
        >>> floats = util.read_first_line_floats('data.txt')
        >>> # Read comma-separated floats
        >>> floats = util.read_first_line_floats('data.csv', delimiter=',')
        """
        try:
            with open(file_path, "r") as fp:
                first_line = fp.readline()

                if not first_line:
                    self.debug(f"File {file_path} is empty")
                    return []

                # Strip specified characters (default: whitespace)
                if strip_chars is not None:
                    first_line = first_line.strip(strip_chars)
                else:
                    first_line = first_line.strip()

                if not first_line:
                    self.debug(f"First line of {file_path} is empty after stripping")
                    return []

                # Auto-detect delimiter if not specified
                if delimiter is None:
                    # Try common delimiters in order of preference
                    for test_delimiter in [" ", ",", "\t", ";", "|"]:
                        if test_delimiter in first_line:
                            delimiter = test_delimiter
                            break
                    else:
                        # No delimiter found, treat as single value
                        delimiter = None

                # Split the line and convert to floats
                if delimiter is not None:
                    string_values = first_line.split(delimiter)
                else:
                    string_values = [first_line]

                # Convert to floats, handling empty strings
                float_values = []
                for value in string_values:
                    value = value.strip()
                    if value:  # Skip empty strings
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

    def df_to_cont_dict(self, df, categorical_column, value_column):
        """Convert a DataFrame with two continuous columns into a dictionary with column names as keys"""
        pass

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
        >>> # Returns: ({'happy': [0.8, 0.9], 'sad': [0.6, 0.5], 'angry': [0.7]}, 1.67)
        """
        if categorical_column not in df.columns:
            self.error(f"Column '{categorical_column}' not found in DataFrame")
        if value_column not in df.columns:
            self.error(f"Column '{value_column}' not found in DataFrame")

        result = {}
        for category in df[categorical_column].unique():
            mask = df[categorical_column] == category
            values = df.loc[mask, value_column].tolist()
            result[category] = values

        # Calculate mean number of values per category
        if result:
            total_values = sum(len(values) for values in result.values())
            mean_values_per_category = total_values / len(result)
        else:
            mean_values_per_category = 0.0

        return result, mean_values_per_category

    def is_dict_with_string_values(self, test_dict):
        try:
            return isinstance(test_dict, dict) and all(
                isinstance(v, str) for v in test_dict.values()
            )
        except (ValueError, SyntaxError):
            return False

    def map_labels(self, df, target, mapping):
        # mapping should be a dictionary, the keys might encode lists.
        keys = list(mapping.keys())
        for key in keys:
            # a comma in the key means that the key is a list of labels
            if "," in key:
                # split the key and create a list
                key_list = [k.strip() for k in key.split(",")]
                # create a new mapping for each key
                for k in key_list:
                    mapping[k] = mapping[key]
                # remove the old key
                del mapping[key]
        # ensure string type for the target column
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
        >>> scaled = util.scale_to_range(values, -1, 1)
        >>> # Returns: [-1.0, -0.5, 0.0, 0.5, 1.0]
        """
        values = np.array(values, dtype=float)

        if len(values) == 0:
            return values

        old_min = np.min(values)
        old_max = np.max(values)

        # Handle case where all values are the same
        if old_min == old_max:
            return np.full_like(values, (new_min + new_max) / 2)

        # Apply min-max scaling formula
        scaled = (values - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

        return scaled
