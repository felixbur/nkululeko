# util.py
import ast
import configparser
import datetime
import logging
import os.path
import shutil
import sys

from pathlib import Path

import numpy as np

import audeer

from nkululeko.utils.dataframe import DataFrameMixin
from nkululeko.utils.naming import NamingMixin
from nkululeko.utils.storage import StorageMixin


class Util(NamingMixin, StorageMixin, DataFrameMixin):
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
        "uar",
        "mse",
    ]
    keyvals = [        
        "kind",
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
        # Always set DEBUG so messages reach all handlers regardless of whether
        # an ancestor logger (e.g. root logger in notebooks) already has handlers.
        logger.setLevel(logging.DEBUG)

        # Create a simple formatter that only shows the message
        class SimpleFormatter(logging.Formatter):
            def format(self, record):
                return record.getMessage()

        # Only add a console handler if this logger has none yet.
        # Use logger.handlers (direct handlers) rather than hasHandlers()
        # so the check is scoped to this logger only, not the full hierarchy.
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(SimpleFormatter())
            logger.addHandler(console_handler)

        # Add or replace file handler when config is available
        if self.config is not None:
            try:
                root = self.config["EXP"]["root"]
                name = self.config["EXP"]["name"]
                log_dir = os.path.abspath(os.path.join(root, name, "log"))
                audeer.mkdir(log_dir)
                # Include seconds to avoid filename collisions between close-together runs
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

                # Collect stale file handlers (different experiment dir) then remove them
                # to avoid mutating the handlers list during iteration.
                stale = [
                    h
                    for h in logger.handlers
                    if isinstance(h, logging.FileHandler)
                    and os.path.dirname(h.baseFilename) != log_dir
                ]
                for handler in stale:
                    handler.close()
                    logger.removeHandler(handler)

                if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
                    file_handler = logging.FileHandler(log_file)
                    file_handler.setFormatter(SimpleFormatter())
                    logger.addHandler(file_handler)

                    # Save a timestamped config snapshot alongside the log file.
                    # Reads --config from sys.argv so no entry-point changes are needed.
                    # Preserves the original file's extension (format-independent).
                    src = None
                    if "--config" in sys.argv:
                        idx = sys.argv.index("--config")
                        if idx + 1 < len(sys.argv):
                            src = sys.argv[idx + 1]
                    if src and os.path.isfile(src):
                        ext = os.path.splitext(src)[1]
                        config_snapshot = os.path.join(
                            log_dir, f"{name}_{timestamp}{ext}"
                        )
                        shutil.copy2(src, config_snapshot)
            except KeyError:
                logger.debug(
                    "File logging skipped: EXP configuration (root/name) incomplete"
                )
            except OSError as e:
                logger.debug(f"File logging skipped: could not create log file ({e})")

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
                            f"value for {key} not found, using default: {default}"
                        )
                    return default
            if default not in self.stopvals:
                self.debug(f"value for {key} not found, using default: {default}")
            return default

    def set_config(self, config):
        self.config = config
        # self.logged_configs.clear()

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

    def extract_parent_and_name(self, path_str):
        """Extract (parent_dir_name in 2 levels, filename) from a path string."""
        p = Path(path_str)
        return (p.parent.parent.name, p.parent.name, p.name)

    def filter_filepath(self, df_source, df_target):
        df_source_keys = {
           self.extract_parent_and_name(path)
            for path in df_source.index.get_level_values(0)
        }
        df_target = df_target[
            df_target.index.get_level_values(0).map(
                lambda p: self.extract_parent_and_name(p) in df_source_keys
            )
        ]
        return df_target


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
            if default not in self.stopvals and key not in self.keyvals:
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

    def print_best_results(self, best_reports):
        res_dir = self.get_res_dir()
        all = ""
        vals = np.empty(0)
        for report in best_reports:
            all += f"{report.result.test:.4f} "
            vals = np.append(vals, report.result.test)
        file_name = f"{res_dir}{self.get_exp_name()}_runs.txt"

        # For metrics where lower is better (EER, MSE, MAE), show min instead of max
        if self.high_is_good():
            best_val = vals.max()
            best_idx = vals.argmax()
            best_label = "max"
        else:
            best_val = vals.min()
            best_idx = vals.argmin()
            best_label = "min"

        output = (
            f"{all}"
            + f"\nmean: {vals.mean():.4f}, std: {vals.std():.4f}, "
            + f"{best_label}: {best_val:.4f}, {best_label}_index: {best_idx}"
        )
        with open(file_name, "w") as text_file:
            text_file.write(output)
        self.debug(output)

    def append_to_result_file(self, filename, content):
        """Append *content* as a new line to *filename*, creating the file if needed.

        The line is only written if it is not already present in the file.

        Args:
            filename: absolute path to the result text file.
            content: string to append (a newline is added automatically).
        """
        existing = []
        if os.path.isfile(filename):
            with open(filename) as f:
                existing = f.read().splitlines()
        if content not in existing:
            with open(filename, "a") as f:
                f.write(content + "\n")

    def check_class_label(self, df):
        target = self.config_val("DATA", "target", None)
        if "class_label" in df.columns and target is not None:
            df = df.drop(columns=[target])
            df = df.rename(columns={"class_label": target})
        return df

    def high_is_good(self):
        """check how to interpret results (higher is better)"""
        if self.exp_is_classification():
            measure = self.config_val("MODEL", "measure", "uar")
            measure_low = ["eer"]
            if measure in measure_low:
                return False
            else:
                return True
        else:
            measure = self.config_val("MODEL", "measure", "mse")
            measure_low = ["mse", "mae"]
            if measure in measure_low:
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
        """Given a float, return this to 3 digits as string with leading zero."""
        return f"{self.to_3_digits(x):.3f}"

    def to_4_digits(self, x):
        """Given a float, return this to 4 digits."""
        x = float(x)
        if np.isnan(x):
            return x
        return (int(x * 10000)) / 10000.0

    def to_4_digits_str(self, x):
        """Given a float, return this to 4 digits as string with leading zero."""
        x_val = self.to_4_digits(x)
        if np.isnan(x_val):
            return "nan"
        return f"{x_val:.4f}"
