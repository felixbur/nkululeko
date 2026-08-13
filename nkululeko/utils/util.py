# util.py
import ast
import configparser
import datetime
import logging
import os.path
import shutil
import sys
import threading
from collections import deque

from pathlib import Path

import audeer
import numpy as np

from nkululeko.utils.dataframe import DataFrameMixin
from nkululeko.utils.errors import NkululukoError
from nkululeko.utils.naming import NamingMixin
from nkululeko.utils.storage import StorageMixin
from nkululeko.experiment_context import get_context

_log_lock = threading.Lock()


class _MessageOnlyFormatter(logging.Formatter):
    def format(self, record):
        return record.getMessage()


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
        "nan_strategy",
    ]

    def __init__(self, caller=None, has_config=True, context=None):
        """Create a utility helper bound to an experiment context.

        ``context`` is optional only for backwards compatibility with callers
        outside the experiment orchestration layer.
        """
        self.logger = None
        self.context = context if context is not None else get_context()
        if caller is not None:
            self.caller = caller
        else:
            self.caller = ""
        self.config = None
        if has_config:
            try:
                self.config = self.context.config
                self.got_data_roots = self.config_val("DATA", "root_folders", False)
                if self.got_data_roots:
                    # if there is a global data rootfolder file, read from
                    # there
                    if not os.path.isfile(self.got_data_roots):
                        self.error(f"no such file: {self.got_data_roots}")
                    self.data_roots = configparser.ConfigParser()
                    self.data_roots.read(self.got_data_roots)
            except AttributeError as e:
                self.error(e)
                self.config = None
                self.got_data_roots = False

        self.setup_logging()
        # self.logged_configs = set()

    def setup_logging(self):
        """Configure and attach this Util instance's logger.

        Ensures a console handler is present, and additionally attaches a
        file handler under the experiment's ``EXP.root``/``EXP.name`` log
        directory if a config is set. Safe to call multiple times: existing
        handlers are reused rather than duplicated.
        """
        logger = logging.getLogger(__name__)
        # Always set DEBUG so messages reach all handlers regardless of whether
        # an ancestor logger (e.g. root logger in notebooks) already has handlers.
        logger.setLevel(logging.DEBUG)
        formatter = _MessageOnlyFormatter()
        self._ensure_console_handler(logger, formatter)

        if self.config is not None:
            self._setup_file_logging(logger, formatter)

        self.logger = logger

    @staticmethod
    def _ensure_console_handler(logger, formatter):
        # Only add a console handler if this logger has none yet.
        # Use logger.handlers (direct handlers) rather than hasHandlers()
        # so the check is scoped to this logger only, not the full hierarchy.
        # _log_lock makes the check-then-act atomic so concurrent
        # threads/instances can't both pass the check and add a handler each.
        with _log_lock:
            if not logger.handlers:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

    def _setup_file_logging(self, logger, formatter):
        try:
            root = self.config["EXP"]["root"]
            name = self.config["EXP"]["name"]
            log_dir, log_file, timestamp = self._build_log_path(root, name)

            # Only the handler check-then-act is locked; the filesystem work
            # above (mkdir) and the config snapshot copy below stay outside
            # the critical section so unrelated threads aren't blocked on I/O.
            with _log_lock:
                self._remove_stale_file_handlers(logger, log_dir)
                if self._has_file_handler(logger):
                    return
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            self._copy_config_snapshot(log_dir, name, timestamp)
        except KeyError:
            logger.debug("File logging skipped: EXP configuration (root/name) incomplete")
        except OSError as e:
            logger.debug(f"File logging skipped: could not create log file ({e})")

    @staticmethod
    def _build_log_path(root, name):
        log_dir = os.path.abspath(os.path.join(root, name, "log"))
        audeer.mkdir(log_dir)
        # Include seconds to avoid filename collisions between close-together runs
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return log_dir, os.path.join(log_dir, f"{name}_{timestamp}.log"), timestamp

    @staticmethod
    def _remove_stale_file_handlers(logger, log_dir):
        stale = [
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
            and os.path.dirname(handler.baseFilename) != log_dir
        ]
        for handler in stale:
            handler.close()
            logger.removeHandler(handler)

    @staticmethod
    def _has_file_handler(logger):
        return any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)

    @staticmethod
    def _config_snapshot_source():
        if "--config" not in sys.argv:
            return None
        idx = sys.argv.index("--config")
        if idx + 1 >= len(sys.argv):
            return None
        return sys.argv[idx + 1]

    def _copy_config_snapshot(self, log_dir, name, timestamp):
        src = self._config_snapshot_source()
        if not src or not os.path.isfile(src):
            return
        ext = os.path.splitext(src)[1]
        config_snapshot = os.path.join(log_dir, f"{name}_{timestamp}{ext}")
        shutil.copy2(src, config_snapshot)

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
        configuration = self.config
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
        """Replace the active experiment configuration.

        Args:
            config: A ``configparser.ConfigParser`` (or compatible mapping) to
                use for subsequent ``config_val*`` lookups.
        """
        self.config = config
        self.context.config = config
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
        """Get the experiment's results directory, creating it if necessary.

        Returns:
            str: Path to ``<exp_dir>/results/``, ending with a slash.
        """
        home_dir = self.get_exp_dir()
        dir_name = f"{home_dir}/results/"
        audeer.mkdir(dir_name)
        return dir_name

    def exp_is_classification(self):
        """Check whether the current experiment is a classification task.

        Reads ``EXP.type`` from the config, defaulting to ``"classification"``.

        Returns:
            bool: True if ``EXP.type`` is ``"classification"``, False otherwise
            (e.g. ``"regression"``).
        """
        type = self.config_val("EXP", "type", "classification")
        if type == "classification":
            return True
        return False

    def error(self, message):
        """Log *message* at ERROR level and raise NkululukoError.

        Args:
            message: The error message to log and raise.

        Raises:
            NkululukoError: Always raised, with ``message`` prefixed by the
                caller name (``self.caller``).
        """
        full_msg = f"ERROR: {self.caller}: {message}"
        if self.logger is not None:
            self.logger.error(full_msg)
        else:
            print(full_msg)
        raise NkululukoError(full_msg)

    def warn(self, message):
        """Log *message* at WARNING level, prefixed by the caller name.

        Args:
            message: The warning message to log.
        """
        if self.logger is not None:
            self.logger.warning(f"WARNING: {self.caller}: {message}")
        else:
            print(f"WARNING: {message}", flush=True)

    def debug(self, message):
        """Log *message* at DEBUG level, prefixed by the caller name.

        Args:
            message: The debug message to log.
        """
        if self.logger is not None:
            self.logger.debug(f"DEBUG: {self.caller}: {message}")
        else:
            print(f"DEBUG: {message}", flush=True)

    def handle_nan(self, df, context="features", strategy=None, allow_drop=True):
        """Handle NaN values in a DataFrame with configurable strategy.

        Args:
            df: pandas DataFrame to check and fill NaN values in.
            context: string describing where the NaN was found (for logging).
            strategy: optional strategy override. If unset, FEATS.nan_strategy is used.
            allow_drop: whether the drop strategy may remove rows.

        Returns:
            DataFrame with NaN values handled according to configured strategy.
        """
        if not df.isna().to_numpy().any():
            return df

        nan_count = df.isna().sum().sum()
        nan_pct = 100 * nan_count / df.size
        raw_strategy = (
            strategy
            if strategy is not None
            else self.config_val("FEATS", "nan_strategy", "zero")
        )
        strategy = str(raw_strategy).strip().lower()
        valid_strategies = {"zero", "mean", "median", "drop"}
        if strategy not in valid_strategies:
            self.warn(
                f"{context}: unknown NaN strategy '{raw_strategy}', using strategy 'zero'"
            )
            strategy = "zero"
        elif strategy == "drop" and not allow_drop:
            self.warn(
                f"{context}: NaN strategy 'drop' is not allowed because it can "
                "misalign features and labels, using strategy 'zero'"
            )
            strategy = "zero"

        self.warn(
            f"{context}: replacing {nan_count} NaN values"
            f" ({nan_pct:.1f}% of data) with strategy '{strategy}'"
        )

        if strategy == "mean":
            # Second fillna(0) handles columns where all values are NaN (mean is NaN)
            numeric_means = df.mean(numeric_only=True)
            return df.fillna(numeric_means).fillna(0)
        elif strategy == "median":
            # Second fillna(0) handles columns where all values are NaN (median is NaN)
            numeric_medians = df.median(numeric_only=True)
            return df.fillna(numeric_medians).fillna(0)
        elif strategy == "drop":
            return df.dropna()
        else:
            # Default: zero
            return df.fillna(0)

    def set_config_val(self, section, key, value):
        """Set a value in the experiment configuration, creating the section if needed.

        Args:
            section: INI section name (e.g. ``"MODEL"``, ``"FEATS"``).
            key: Key within the section.
            value: Value to store; converted to ``str`` before writing.
        """
        try:
            # does the section already exists?
            self.config[section][key] = str(value)
        except KeyError:
            self.config.add_section(section)
            self.config[section][key] = str(value)

    def exists_config_val(self, section, key):
        """Check whether a key is present in the experiment configuration.

        Args:
            section: INI section name (e.g. ``"MODEL"``, ``"FEATS"``).
            key: Key within the section.

        Returns:
            bool: True if the section and key both exist, False otherwise.
        """
        try:
            _ = self.config[section][key]
            return True
        except KeyError:
            return False

    def extract_parent_and_name(self, path_str):
        """Extract (parent_dir_name in 2 levels, filename) from a path string."""
        p = Path(path_str)
        return (p.parent.parent.name, p.parent.name, p.name)

    def filter_filepath(self, df_source, df_target):
        """Restrict df_target to rows whose file path -- and, for a
        segmented (file, start, end) index, the exact start/end too --
        also occurs in df_source, returned IN df_source'S ROW ORDER.

        Callers use this to align two dataframes describing the same
        samples (e.g. extracted features against their labels) that are
        then consumed *positionally* -- row i of one paired with row i of
        the other (see e.g. MLPModel.get_loader, which does
        ``df_x.values[i]`` next to ``df_y[...].iloc[i]``). df_target's own
        row order generally has nothing to do with df_source's (e.g.
        features come out in extraction order, a split's labels come out
        in whatever order the split/concat produced, or a cached split
        reloaded from CSV) -- returning df_target filtered but still in
        *its own* order would silently pair each row with the wrong
        label/feature the moment the two orders diverge. Reordering to
        match df_source removes that whole class of bug rather than
        relying on the two orders happening to already coincide.

        The file-path component is matched on ``(grandparent dir name,
        parent dir name, file name)`` via :meth:`extract_parent_and_name`,
        rather than the full path, so this is robust to different
        absolute-path prefixes for what's otherwise the same file. Any
        additional index levels (segment start/end) are matched exactly,
        so a file with multiple segments/utterances isn't over-matched:
        only the *specific* segments present in df_source are kept, not
        every segment of any file df_source happens to touch.

        Args:
            df_source: DataFrame whose index (and order) provides the
                (file[, start, end]) keys to keep, and the row order of
                the result.
            df_target: DataFrame to filter and reorder.

        Returns:
            DataFrame: the subset of df_target matching df_source, one row
            per matched df_source row, in df_source's row order. A
            df_source row with no match in df_target is skipped.
        """

        def _key(index_entry):
            if isinstance(index_entry, tuple):
                path, *rest = index_entry
                return (self.extract_parent_and_name(path), *rest)
            return (self.extract_parent_and_name(index_entry),)

        target_positions_by_key = {}
        for pos, idx in enumerate(df_target.index):
            target_positions_by_key.setdefault(_key(idx), deque()).append(pos)

        positions = []
        for idx in df_source.index:
            matches = target_positions_by_key.get(_key(idx))
            if matches:
                positions.append(matches.popleft())

        return df_target.iloc[positions]


    def check_df(self, i, df):
        """Check a dataframe."""
        print(f"check {i}: {df.shape}")
        print(df.head(1))

    def config_val(self, section, key, default):
        """Get a value from the experiment config with a fallback default.

        The most-used lookup method in the codebase; other ``config_val_*``
        helpers build on it to add type coercion or list/dataset semantics.

        Args:
            section: INI section name (e.g. ``"MODEL"``, ``"FEATS"``).
            key: Key within the section.
            default: Value to return if the section/key is not present, or if
                no config has been loaded at all.

        Returns:
            The config value as a string, or ``default`` if not set.
        """
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
        """Clear the set of already-logged default-value warnings.

        Intended to let a fresh experiment run re-log "using default" debug
        messages that a previous run in the same process already logged.

        No-op if ``logged_configs`` was never initialized (the corresponding
        ``self.logged_configs = set()`` in ``__init__`` is commented out).
        """
        if hasattr(cls, "logged_configs"):
            cls.logged_configs.clear()

    def config_val_bool(self, section, key, default=False):
        """Get a boolean configuration value safely without using eval().

        Args:
            section: The config section name.
            key: The config key name.
            default: The default value (bool or string).

        Returns:
            bool: The boolean value of the config entry.
        """
        val = self.config_val(section, key, str(default))
        return str(val).strip().lower() in ("true", "1", "yes")

    def config_val_list(self, section, key, default):
        """Get a config value parsed as a Python literal (e.g. a list or tuple).

        The stored string is evaluated with ``ast.literal_eval``, so it must
        be a valid Python literal (e.g. ``"['os', 'praat']"``).

        Args:
            section: INI section name (e.g. ``"MODEL"``, ``"FEATS"``).
            key: Key within the section.
            default: Value to return if the key is not present.

        Returns:
            The parsed Python object, or ``default`` if not set.
        """
        try:
            return ast.literal_eval(self.config[section][key])
        except KeyError:
            if default not in self.stopvals:
                self.debug(f"value for {key} not found, using default: {default}")
            return default

    def print_best_results(self, best_reports):
        """Summarize and write the best result of each run to a text file.

        Computes the mean, std, and best (max or min, depending on
        :meth:`high_is_good`) test score across ``best_reports``, writes the
        summary to ``<res_dir>/<exp_name>_runs.txt``, and logs it via debug.

        Args:
            best_reports: Iterable of report objects, each exposing
                ``report.result.test`` as the test-set score for that run.
        """
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
        """Restore the original target column name from a class_label backup.

        If df has a ``class_label`` column (the pre-encoding backup of the
        target values, see CONTRIBUTING.md) and a target is configured, the
        current (possibly integer-encoded) target column is dropped and
        ``class_label`` is renamed back to the target name.

        Args:
            df: DataFrame to check, potentially containing a class_label column.

        Returns:
            DataFrame: df unchanged, or with class_label renamed to the target
            column name.
        """
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
        return f"{x:.3f}"

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
