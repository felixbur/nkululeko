# nkululeko/experiment.py: Main class for an experiment (nkululeko.nkululeko)
import ast
import os
import pickle
import re
import time

import audeer
import audformat
import pandas as pd

import nkululeko.glob_conf as glob_conf
from nkululeko.data.dataset import Dataset
from nkululeko.data.dataset_csv import Dataset_CSV
from nkululeko.data.datasplitter import Datasplitter
from nkululeko.demo_predictor import Demo_predictor
from nkululeko.feat_extract.feats_analyser import FeatureAnalyser
from nkululeko.feature_extractor import FeatureExtractor
from nkululeko.plots import Plots
from nkululeko.reporting.report import Report
from nkululeko.runmanager import Runmanager
from nkululeko.scaler import Scaler
from nkululeko.testing_predictor import TestPredictor
from nkululeko.utils.util import Util


class Experiment:
    """Main class specifying an experiment."""

    def __init__(self, config_obj):
        """Constructor.

        Args:
            - config_obj : a config parser object that sets the experiment parameters and being set as a global object.
        """
        self.set_globals(config_obj)
        self.name = glob_conf.config["EXP"]["name"]
        self.root = os.path.join(glob_conf.config["EXP"]["root"], "")
        self.data_dir = os.path.join(self.root, self.name)
        audeer.mkdir(self.data_dir)  # create the experiment directory
        self.util = Util("experiment")
        glob_conf.set_util(self.util)
        self.split3 = eval(self.util.config_val("EXP", "traindevtest", "False"))
        glob_conf.set_split3(self.split3)
        fresh_report = eval(self.util.config_val("REPORT", "fresh", "False"))
        if not fresh_report:
            try:
                with open(os.path.join(self.data_dir, "report.pkl"), "rb") as handle:
                    self.report = pickle.load(handle)
            except FileNotFoundError:
                self.report = Report()
        else:
            self.util.debug("starting a fresh report")
            self.report = Report()
        glob_conf.set_report(self.report)
        self.loso = self.util.config_val("MODEL", "loso", False)
        self.logo = self.util.config_val("MODEL", "logo", False)
        self.xfoldx = self.util.config_val("MODEL", "k_fold_cross", False)
        self.start = time.process_time()
        self.df_dev = None
        self.df_train = None
        self.df_test = None
        self.feats_train = None
        self.feats_test = None
        self.feats_dev = None

    def set_module(self, module):
        glob_conf.set_module(module)

    def store_report(self):
        with open(os.path.join(self.data_dir, "report.pkl"), "wb") as handle:
            pickle.dump(self.report, handle)
        if eval(self.util.config_val("REPORT", "show", "False")):
            self.report.print()
        if self.util.config_val("REPORT", "latex", False):
            self.report.export_latex()

    # moved to util
    # def get_name(self):
    #     return self.util.get_exp_name()

    def set_globals(self, config_obj):
        """Install a config object in the global space."""
        glob_conf.init_config(config_obj)

    def load_datasets(self):
        """Load all databases specified in the configuration and map the labels."""
        ds = ast.literal_eval(glob_conf.config["DATA"]["databases"])
        self.datasets = {}
        self.got_speaker, self.got_gender, self.got_age = False, False, False
        for d in ds:
            ds_type = self.util.config_val_data(d, "type", "audformat")
            if ds_type == "audformat":
                data = Dataset(d)
            elif ds_type == "csv":
                data = Dataset_CSV(d)
            else:
                self.util.error(f"unknown data type: {ds_type}")
            data.load()
            data.prepare()
            if data.got_gender:
                self.got_gender = True
            if data.got_age:
                self.got_age = True
            if data.got_speaker:
                self.got_speaker = True
            self.datasets.update({d: data})
        self.target = self.util.config_val("DATA", "target", None)
        glob_conf.set_got_speaker(self.got_speaker)
        glob_conf.set_target(self.target)
        # print target via debug
        self.util.debug(f"target: {self.target}")
        # print keys/column
        dbs = ",".join(list(self.datasets.keys()))
        # Support both None and 'none' for backward compatibility
        if self.target is None or self.target == "none":
            self.util.debug(f"loaded databases {dbs}")
            return
        labels = self.util.config_val("DATA", "labels", False)
        auto_labels = list(next(iter(self.datasets.values())).df[self.target].unique())
        if labels:
            self.labels = ast.literal_eval(labels)
            self.util.debug(f"Using target labels (from config): {labels}")
        else:
            self.labels = auto_labels
        # print autolabel no matter it is specified or not
        self.util.debug(f"Labels (from database): {auto_labels}")
        glob_conf.set_labels(self.labels)
        self.util.debug(f"loaded databases {dbs}")
        self.datasplitter = Datasplitter(self.datasets)

    def _import_csv(self, storage):
        # df = pd.read_csv(storage, header=0, index_col=[0,1,2])
        # df.index.set_levels(pd.to_timedelta(df.index.levels[1]), level=1)
        # df.index.set_levels(pd.to_timedelta(df.index.levels[2]), level=2)
        try:
            df = audformat.utils.read_csv(storage)
        except ValueError:
            # split might be empty
            return pd.DataFrame()
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, pd.Index):
            df = pd.DataFrame(index=df)
        df.is_labeled = True if self.target in df else False
        # print(df.head())
        return df

    def fill_tests(self, encode=True):
        """Only fill a new test set.

        Args:
            encode: When True (default), integer-encode the target column using
                the training label encoder and cache the result.  Pass False to
                keep original string labels (the caller is then responsible for
                encoding before passing the dataframe to a model).
        """
        test_dbs = ast.literal_eval(glob_conf.config["DATA"]["tests"])
        self.df_test = pd.DataFrame()
        start_fresh = eval(self.util.config_val("DATA", "no_reuse", "False"))
        store = self.util.get_path("store")
        storage_test = f"{store}extra_testdf.csv"
        # Only use the cached (integer-encoded) CSV when encode=True
        if encode and os.path.isfile(storage_test) and not start_fresh:
            self.util.debug(f"reusing previously stored {storage_test}")
            self.df_test = self._import_csv(storage_test)
        else:
            for d in test_dbs:
                ds_type = self.util.config_val_data(d, "type", "audformat")
                if ds_type == "audformat":
                    data = Dataset(d)
                elif ds_type == "csv":
                    data = Dataset_CSV(d)
                else:
                    self.util.error(f"unknown data type: {ds_type}")
                data.load()
                if data.got_gender:
                    self.got_gender = True
                if data.got_age:
                    self.got_age = True
                if data.got_speaker:
                    self.got_speaker = True
                data.split()
                data.prepare_labels()
                self.df_test = pd.concat(
                    [self.df_test, self.util.make_segmented_index(data.df_test)]
                )
                self.df_test.is_labeled = data.is_labeled
            self.df_test.got_gender = self.got_gender
            self.df_test.got_speaker = self.got_speaker
            if encode:
                self.df_test["class_label"] = self.df_test[self.target]
                self.df_test[self.target] = self.label_encoder.transform(
                    self.df_test[self.target]
                )
                self.df_test.to_csv(storage_test)

    def fill_train_and_tests(self):
        if self.split3:
            self.df_train, self.df_test, self.df_dev = (
                self.datasplitter.fill_train_and_tests()
            )
        else:
            self.df_train, self.df_test = self.datasplitter.fill_train_and_tests()
        self.test_ds_df = getattr(self.datasplitter, "test_ds_df", {})
        self.label_encoder = glob_conf.label_encoder

    def extract_test_feats(self):
        self.feats_test = pd.DataFrame()
        feats_name = "_".join(ast.literal_eval(glob_conf.config["DATA"]["tests"]))
        feats_types = self.util.config_val_list("FEATS", "type", ["os"])
        self.feature_extractor = FeatureExtractor(
            self.df_test, feats_types, feats_name, "test"
        )
        self.feats_test = self.feature_extractor.extract()
        self.util.debug(f"Test features shape:{self.feats_test.shape}")

    def evaluate_per_test_set(self):
        """Evaluate the best model on each test dataset individually.

        When multiple datasets contribute test samples, this method evaluates
        the best trained model on each dataset separately and reports the
        per-dataset results (confusion matrix, UAR/MSE, …).  It is a no-op
        when fewer than two test datasets are present.
        """
        if not hasattr(self, "test_ds_df") or len(self.test_ds_df) <= 1:
            return
        if self.datasplitter.df_test.empty:
            return
        if not hasattr(self, "runmgr") or not hasattr(self.runmgr, "modelrunner"):
            return
        self.util.debug(
            f"Evaluating {len(self.test_ds_df)} test datasets individually…"
        )
        best_model = self.runmgr.get_best_model()
        plot_name_suggest = self.util.get_exp_name()
        for ds_name, df_test_ds in self.test_ds_df.items():
            feats_test_ds = self.util.filter_filepath(df_test_ds, self.feats_test)
            # feats_test_ds = self.feats_test[
            #     self.feats_test.index.isin(df_test_ds.index)
            # ]
            if feats_test_ds.shape[0] == 0:
                self.util.warn(f"{ds_name}: no features found for test set, skipping")
                continue
            df_test_aligned = self.util.filter_filepath(feats_test_ds, df_test_ds)
            # df_test_aligned = df_test_ds[df_test_ds.index.isin(feats_test_ds.index)]
            if df_test_aligned.shape[0] == 0:
                self.util.warn(
                    f"{ds_name}: no samples after alignment with features, skipping"
                )
                continue
            self.util.debug(
                f"Evaluating on test dataset {ds_name}:"
                f" {df_test_aligned.shape[0]} samples"
            )
            report = self.runmgr.modelrunner.eval_specific_model(
                best_model, df_test_aligned, feats_test_ds
            )
            safe_name = re.sub(r"[^\w-]", "_", ds_name)
            plot_name = (
                self.util.config_val("PLOT", "name", plot_name_suggest)
                + f"_{safe_name}_test"
            )
            self.runmgr.print_report(report, plot_name)

    def _decode_labels(self, df_labels, column_name):
        """Decode encoded labels for visualization.

        Args:
            df_labels: DataFrame containing the labels
            column_name: Name of the column to decode

        Returns:
            str: The column name to use (either decoded version or original)
        """
        if (
            hasattr(self, "label_encoder")
            and self.label_encoder is not None
            and self.util.exp_is_classification()
        ):
            decoded_col = f"{column_name}_decoded"
            df_labels[decoded_col] = self.label_encoder.inverse_transform(
                df_labels[column_name]
            )
            return decoded_col
        return column_name

    def plot_distribution(self, df_labels):
        """Plot the distribution of samples and speakers.

        Per target class and biological sex.
        """
        plot = Plots()
        plot.plot_distributions(df_labels)
        if self.got_speaker:
            plot.plot_distributions_speaker(df_labels)

    def extract_feats(self):
        """Extract the features for train, test and dev sets.

        They will be stored on disk and need to be removed manually.

        The string FEATS.feats_type is read from the config, defaults to os.

        """
        if self.split3:
            self.feats_train, self.feats_test, self.feats_dev = (
                self.datasplitter.extract_feats()
            )
        else:
            self.feats_train, self.feats_test = self.datasplitter.extract_feats()
        if self.feats_train is None:
            return
        # sync back dataframes to the experiment object, as they might have changed
        self.df_train = self.datasplitter.df_train
        self.df_test = self.datasplitter.df_test
        if self.split3:
            self.df_dev = self.datasplitter.df_dev
        self._check_scale()

    def get_sample_selection(self):
        """Return the configured sample selection.

        This delegates to ``Datasplitter.get_sample_selection()`` to preserve
        the previous ``Experiment`` API expected by existing callers.
        """
        return self.datasplitter.get_sample_selection()

    def augment(self, method="audiomentations"):
        """Augment the selected samples."""
        sample_selection = self.util.config_val("DATA", "sample_selection", "all")
        if sample_selection == "all":
            df = pd.concat([self.df_train, self.df_test])
        elif sample_selection == "train":
            df = self.df_train
        elif sample_selection == "test":
            df = self.df_test
        else:
            self.util.error(
                f"unknown augmentation selection specifier {sample_selection},"
                " should be [all | train | test]"
            )
        if method == "audiomentations":
            from nkululeko.augmenting.augmenter_audiomentations import (
                AugmenterAudiomentations,
            )

            augmenter = AugmenterAudiomentations(df)
        elif method == "auglib":
            from nkululeko.augmenting.augmenter_auglib import AugmenterAuglib

            augmenter = AugmenterAuglib(df)
        else:
            self.util.error(f"unknown augmentation method: {method}")

        df_ret = augmenter.augment(sample_selection)
        return df_ret

    def autopredict(self):
        """Predict labels for samples with existing models and add to the dataframe."""
        sample_selection = self.util.config_val("PREDICT", "sample_selection", "all")
        if sample_selection == "all":
            df = pd.concat([self.df_train, self.df_test])
        elif sample_selection == "train":
            df = self.df_train
        elif sample_selection == "test":
            df = self.df_test
        else:
            self.util.error(
                f"unknown augmentation selection specifier {sample_selection},"
                " should be [all | train | test]"
            )
        targets = self.util.config_val_list("PREDICT", "targets", None)
        if targets is None:
            self.util.error("no prediction target specified")
        for target in targets:
            if target == "speaker":
                from nkululeko.autopredict.ap_sid import SIDPredictor

                predictor = SIDPredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "gender":
                from nkululeko.autopredict.ap_gender import GenderPredictor

                predictor = GenderPredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "age":
                from nkululeko.autopredict.ap_age import AgePredictor

                predictor = AgePredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "snr":
                from nkululeko.autopredict.ap_snr import SNRPredictor

                predictor = SNRPredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "mos":
                from nkululeko.autopredict.ap_mos import MOSPredictor

                predictor = MOSPredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "pesq":
                from nkululeko.autopredict.ap_pesq import PESQPredictor

                predictor = PESQPredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "sdr":
                from nkululeko.autopredict.ap_sdr import SDRPredictor

                predictor = SDRPredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "stoi":
                from nkululeko.autopredict.ap_stoi import STOIPredictor

                predictor = STOIPredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "text":
                from nkululeko.autopredict.ap_text import TextPredictor

                predictor = TextPredictor(df, self.util)
                df = predictor.predict(sample_selection)
            elif target == "textclassification":
                from nkululeko.autopredict.ap_textclassifier import (
                    TextClassificationPredictor,
                )

                predictor = TextClassificationPredictor(df, self.util)
                df = predictor.predict(sample_selection)
            elif target == "translation":
                from nkululeko.autopredict.ap_translate import TextTranslator

                predictor = TextTranslator(df, self.util)
                df = predictor.predict(sample_selection)
            elif target == "arousal":
                from nkululeko.autopredict.ap_arousal import ArousalPredictor

                predictor = ArousalPredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "valence":
                from nkululeko.autopredict.ap_valence import ValencePredictor

                predictor = ValencePredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "dominance":
                from nkululeko.autopredict.ap_dominance import DominancePredictor

                predictor = DominancePredictor(df)
                df = predictor.predict(sample_selection)
            elif target == "emotion":
                from nkululeko.autopredict.ap_emotion import EmotionPredictor

                predictor = EmotionPredictor(df)
                df = predictor.predict(sample_selection)
            else:
                self.util.error(f"unknown auto predict target: {target}")
        return df

    def random_splice(self):
        """
        Random-splice the selected samples
        """
        from nkululeko.augmenting.randomsplicer import Randomsplicer

        sample_selection = self.util.config_val("EXP", "sample_selection", "train")
        if sample_selection == "all":
            df = pd.concat([self.df_train, self.df_test])
        elif sample_selection == "train":
            df = self.df_train
        elif sample_selection == "test":
            df = self.df_test
        else:
            self.util.error(
                f"unknown augmentation selection specifier {sample_selection},"
                " should be [all | train | test]"
            )
        randomsplicer = Randomsplicer(df)
        df_ret = randomsplicer.run(sample_selection)
        return df_ret

    def analyse_features(self, needs_feats):
        """Do a feature exploration."""
        plot_feats = eval(
            self.util.config_val("EXPL", "feature_distributions", "False")
        )
        sample_selection = self.util.config_val("EXP", "sample_selection", "all")
        # get the data labels
        if sample_selection == "all":
            df_labels = pd.concat([self.df_train, self.df_test])
            self.util.copy_flags(self.df_train, df_labels)
        elif sample_selection == "train":
            df_labels = self.df_train
            self.util.copy_flags(self.df_train, df_labels)
        elif sample_selection == "test":
            df_labels = self.df_test
            self.util.copy_flags(self.df_test, df_labels)
        else:
            self.util.error(
                f"unknown sample selection specifier {sample_selection}, should"
                " be [all | train | test]"
            )
        self.util.debug(f"sampling selection: {sample_selection}")
        if self.util.config_val("EXPL", "value_counts", False):
            self.plot_distribution(df_labels)
        print_colvals = eval(self.util.config_val("EXPL", "print_colvals", "False"))
        if print_colvals:
            self.util.debug(f"columns in data: {df_labels.columns}")
            for col in df_labels.columns:
                self.util.debug(f"{col}: {df_labels[col].unique()}")

        # check if data should be shown with the spotlight data visualizer
        spotlight = eval(self.util.config_val("EXPL", "spotlight", "False"))
        if spotlight:
            self.util.debug("opening spotlight tab in web browser")
            from renumics import spotlight

            spotlight.show(df_labels.reset_index())

        if not needs_feats:
            return
        # get the feature values
        if sample_selection == "all":
            df_feats = pd.concat([self.feats_train, self.feats_test])
        elif sample_selection == "train":
            df_feats = self.feats_train
        elif sample_selection == "test":
            df_feats = self.feats_test
        else:
            self.util.error(
                f"unknown sample selection specifier {sample_selection}, should"
                " be [all | train | test]"
            )
        feat_analyser = FeatureAnalyser(sample_selection, df_labels, df_feats)
        # check if SHAP features should be analysed
        shap = eval(self.util.config_val("EXPL", "shap", "False"))
        if shap:
            feat_analyser.analyse_shap(self.runmgr.get_best_model())

        if plot_feats:
            feat_analyser.analyse()

        # check if a scatterplot should be done
        list_of_dimreds = eval(self.util.config_val("EXPL", "scatter", "False"))

        # Priority: use [EXPL][scatter.target] if available, otherwise use [DATA][target] value
        if (
            hasattr(self, "target")
            and self.target is not None
            and self.target != "none"
        ):
            default_scatter_target = f"['{self.target}']"
        else:
            default_scatter_target = "['class_label']"

        scatter_target = self.util.config_val(
            "EXPL", "scatter.target", default_scatter_target
        )

        if scatter_target == default_scatter_target:
            self.util.debug(
                f"scatter.target using default from [DATA][target]: {scatter_target}"
            )
        else:
            self.util.debug(
                f"scatter.target from [EXPL][scatter.target]: {scatter_target}"
            )
        if list_of_dimreds:
            dimreds = list_of_dimreds
            scat_targets = ast.literal_eval(scatter_target)
            plots = Plots()
            for scat_target in scat_targets:
                # Check if this is the target column that was label-encoded
                is_encoded_target = (
                    scat_target == self.target
                    and hasattr(self, "label_encoder")
                    and self.label_encoder is not None
                    and self.util.exp_is_classification()
                )

                if is_encoded_target:
                    # Decode the labels for visualization
                    target_col = self._decode_labels(df_labels, scat_target)
                    for dimred in dimreds:
                        plots.scatter_plot(df_feats, df_labels, target_col, dimred)
                elif self.util.is_categorical(df_labels[scat_target]):
                    for dimred in dimreds:
                        plots.scatter_plot(df_feats, df_labels, scat_target, dimred)
                else:
                    self.util.debug(
                        f"{self.name}: binning continuous variable to categories"
                    )
                    cat_vals = self.util.continuous_to_categorical(
                        df_labels[scat_target]
                    )
                    df_labels[f"{scat_target}_bins"] = cat_vals.values
                    for dimred in dimreds:
                        plots.scatter_plot(
                            df_feats, df_labels, f"{scat_target}_bins", dimred
                        )

        # check if t-SNE plot should be generated
        tsne = eval(self.util.config_val("EXPL", "tsne", "False"))
        if tsne:
            target_column = self.util.config_val("DATA", "target", "emotion")
            # Decode labels if they were encoded
            target_column = self._decode_labels(df_labels, target_column)
            plots = Plots()
            self.util.debug("generating t-SNE plot...")
            plots.scatter_plot(df_feats, df_labels, target_column, "tsne")

        # check if UMAP plot should be generated
        umap_plot = eval(self.util.config_val("EXPL", "umap", "False"))
        if umap_plot:
            target_column = self.util.config_val("DATA", "target", "emotion")
            # Decode labels if they were encoded
            target_column = self._decode_labels(df_labels, target_column)
            plots = Plots()
            self.util.debug("generating UMAP plot...")
            plots.scatter_plot(df_feats, df_labels, target_column, "umap")

        # check if PCA plot should be generated
        pca_plot = eval(self.util.config_val("EXPL", "pca", "False"))
        if pca_plot:
            target_column = self.util.config_val("DATA", "target", "emotion")
            # Decode labels if they were encoded
            target_column = self._decode_labels(df_labels, target_column)
            plots = Plots()
            self.util.debug("generating PCA plot...")
            plots.scatter_plot(df_feats, df_labels, target_column, "pca")

    def _check_scale(self):
        self.util.save_to_store(self.feats_train, "feats_train")
        self.util.save_to_store(self.feats_test, "feats_test")
        if self.split3:
            self.util.save_to_store(self.feats_dev, "feats_dev")
        scale_feats = self.util.config_val("FEATS", "scale", False)
        # print the scale
        self.util.debug(f"scaler: {scale_feats}")
        if scale_feats:
            self.scaler_feats = Scaler(
                self.df_train,
                self.df_test,
                self.feats_train,
                self.feats_test,
                scale_feats,
                dev_x=self.df_dev,
                dev_y=self.feats_dev,
            )
            if self.split3:
                self.feats_train, self.feats_dev, self.feats_test = (
                    self.scaler_feats.scale()
                )
                # store versions
                self.util.save_to_store(self.feats_train, "feats_train_scaled")
                self.util.save_to_store(self.feats_test, "feats_test_scaled")
                self.util.save_to_store(self.feats_dev, "feats_dev_scaled")
            else:
                self.feats_train, self.feats_test = self.scaler_feats.scale()
                # store versions
                self.util.save_to_store(self.feats_train, "feats_train_scaled")
                self.util.save_to_store(self.feats_test, "feats_test_scaled")

    def init_runmanager(self):
        """Initialize the manager object for the runs."""
        if self.split3:
            self.runmgr = Runmanager(
                self.df_train,
                self.df_test,
                self.feats_train,
                self.feats_test,
                dev_x=self.df_dev,
                dev_y=self.feats_dev,
            )
        else:
            self.runmgr = Runmanager(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )

    def run(self):
        """Do the runs."""
        self.runmgr.do_runs()

        # access the best results all runs
        self.reports = self.runmgr.best_results
        last_epochs = self.runmgr.last_epochs
        # try to save yourself
        save = self.util.config_val("EXP", "save", True)
        if save:
            # save the experiment for future use
            self.save(self.util.get_save_name())
            # self.save_onnx(self.util.get_save_name())

        # self.__collect_reports()
        self.util.print_best_results(self.reports)

        # check if the test predictions should be saved to disk
        test_pred_file = self.util.config_val("EXP", "save_test", False)
        if test_pred_file:
            self.predict_test_and_save(test_pred_file)

        # check if the majority voting for all speakers should be plotted
        conf_mat_per_speaker_function = self.util.config_val(
            "PLOT", "combine_per_speaker", False
        )
        if conf_mat_per_speaker_function:
            self.plot_confmat_per_speaker(conf_mat_per_speaker_function)

        # check if a summary of multiple runs should be plotted
        plot_runs = self.util.config_val("PLOT", "runs_compare", False)
        run_num = int(self.util.config_val("EXP", "runs", 1))
        if plot_runs and run_num > 1:
            from nkululeko.reporting.run_plotter import Run_plotter

            rp = Run_plotter(self)
            rp.plot(plot_runs)

        used_time = time.process_time() - self.start
        self.util.debug(f"Done, used {used_time:.3f} seconds")

        # check if a test set should be labeled by the model:
        label_data = self.util.config_val("DATA", "label_data", False)
        label_result = self.util.config_val("DATA", "label_result", False)
        if label_data and label_result:
            self.predict_test_and_save(label_result)

        return self.reports, last_epochs

    def plot_confmat_per_speaker(self, function):
        if self.loso or self.logo or self.xfoldx:
            self.util.debug(
                "plot combined speaker predictions not possible for cross validation"
            )
            return
        best = self.get_best_report(self.reports)
        if best.is_classification:
            truths = best.truths
            preds = best.preds
        else:
            truths = best.truths_cont
            preds = best.preds_cont
        speakers = self.df_test.speaker.values
        df = pd.DataFrame(data={"truths": truths, "preds": preds, "speakers": speakers})
        plot_name = f"{self.util.get_exp_name()}_speakercombined_{function}"
        self.util.debug(
            f"plotting speaker combination ({function}) confusion matrix to {plot_name}"
        )
        best.plot_per_speaker(df, plot_name, function)

    def get_best_report(self, reports):
        return self.runmgr.get_best_result(reports)

    def print_best_model(self):
        self.runmgr.print_best_result_runs()

    def demo(self, file, is_list, outfile):
        model = self.runmgr.get_best_model()
        lab_enc = None
        try:
            lab_enc = self.label_encoder
        except AttributeError:
            pass
        demo = Demo_predictor(
            model, file, is_list, self.datasplitter.feature_extractor, lab_enc, outfile
        )
        demo.run_demo()

    def predict_test_and_save(self, result_name):
        model = self.runmgr.get_best_model()
        model.set_testdata(self.df_test, self.feats_test)
        test_predictor = TestPredictor(
            model, self.df_test, self.label_encoder, result_name
        )
        result = test_predictor.predict_and_store()
        return result

    def load(self, filename):
        try:
            f = open(filename, "rb")
            tmp_dict = pickle.load(f)
            f.close()
        except EOFError as eof:
            self.util.error(f"can't open file {filename}: {eof}")
        self.__dict__.update(tmp_dict)
        glob_conf.set_labels(self.labels)

    def save(self, filename):
        if self.runmgr.modelrunner.model.is_ann():
            self.runmgr.modelrunner.model = None
            self.util.warn(
                "Save experiment: Can't pickle the trained model so saving without it. (it should be stored anyway)"
            )
        try:
            f = open(filename, "wb")
            pickle.dump(self.__dict__, f)
            f.close()
        except (TypeError, AttributeError) as error:
            # Strip the un-picklable inner model(s) from every FeatureExtractor
            # stored on the experiment. There are typically two: one set in
            # fill_train_and_tests() and one inside Datasplitter.extract_feats().
            # Both may hold the same kind of feat_extractor (e.g. an ONNX
            # InferenceSession from audwav2vec2 / agender), so nulling just
            # one isn't enough.
            for fe in self._collect_feature_extractors():
                inner = getattr(fe, "feat_extractor", None)
                if inner is None:
                    continue
                if hasattr(inner, "model"):
                    inner.model = None
                if hasattr(inner, "model_interface"):
                    inner.model_interface = None
                if hasattr(inner, "model_loaded"):
                    inner.model_loaded = False
            f = open(filename, "wb")
            pickle.dump(self.__dict__, f)
            f.close()
            self.util.warn(
                "Save experiment: Can't pickle the feature extraction model so saving without it."
                + f"{type(error).__name__} {error}"
            )
        except RuntimeError as error:
            self.util.warn(
                "Save experiment: Can't pickle local object, NOT saving: "
                + f"{type(error).__name__} {error}"
            )

    def _collect_feature_extractors(self):
        """Return every FeatureExtractor reachable from `self`.

        Currently looks at `self.feature_extractor` and
        `self.datasplitter.feature_extractor`. Returns an iterable of objects
        that may have a `feat_extractor` attribute.
        """
        seen = []
        fe = getattr(self, "feature_extractor", None)
        if fe is not None:
            seen.append(fe)
        ds = getattr(self, "datasplitter", None)
        ds_fe = getattr(ds, "feature_extractor", None) if ds is not None else None
        if ds_fe is not None and ds_fe is not fe:
            seen.append(ds_fe)
        return seen

    def save_onnx(self, filename):
        # export the model to onnx
        model = self.runmgr.get_best_model()
        if model.is_ann():
            print("converting to onnx from torch")
        else:
            print("converting to onnx from sklearn")
        # save the rest
        f = open(filename, "wb")
        pickle.dump(self.__dict__, f)
        f.close()
