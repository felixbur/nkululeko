# experiment.py: Main class for an experiment (nkululeko.nkululeko)
import ast
import os
import pickle
import random
import time

import audeer
import audformat
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import nkululeko.glob_conf as glob_conf
from nkululeko.data.dataset import Dataset
from nkululeko.data.dataset_csv import Dataset_CSV
from nkululeko.demo_predictor import Demo_predictor
from nkululeko.feat_extract.feats_analyser import FeatureAnalyser
from nkululeko.feature_extractor import FeatureExtractor
from nkululeko.file_checker import FileChecker
from nkululeko.filter_data import DataFilter, filter_min_dur
from nkululeko.plots import Plots
from nkululeko.reporting.report import Report
from nkululeko.runmanager import Runmanager
from nkululeko.scaler import Scaler
from nkululeko.test_predictor import Test_predictor
from nkululeko.utils.util import Util


class Experiment:
    """Main class specifying an experiment"""

    def __init__(self, config_obj):
        """
        Parameters
        ----------
        config_obj : a config parser object that sets the experiment parameters and being set as a global object.
        """

        self.set_globals(config_obj)
        self.name = glob_conf.config["EXP"]["name"]
        self.root = os.path.join(glob_conf.config["EXP"]["root"], "")
        self.data_dir = os.path.join(self.root, self.name)
        audeer.mkdir(self.data_dir)  # create the experiment directory
        self.util = Util("experiment")
        glob_conf.set_util(self.util)
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

    def set_module(self, module):
        glob_conf.set_module(module)

    def store_report(self):
        with open(os.path.join(self.data_dir, "report.pkl"), "wb") as handle:
            pickle.dump(self.report, handle)
        if eval(self.util.config_val("REPORT", "show", "False")):
            self.report.print()
        if self.util.config_val("REPORT", "latex", False):
            self.report.export_latex()

    def get_name(self):
        return self.util.get_exp_name()

    def set_globals(self, config_obj):
        """install a config object in the global space"""
        glob_conf.init_config(config_obj)

    def load_datasets(self):
        """Load all databases specified in the configuration and map the labels"""
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
        self.target = self.util.config_val("DATA", "target", "emotion")
        # print target via debug
        self.util.debug(f"target: {self.target}")
        # print keys/column
        dbs = ",".join(list(self.datasets.keys()))
        labels = self.util.config_val("DATA", "labels", False)
        if labels:
            self.labels = ast.literal_eval(labels)
            self.util.debug(f"Target labels (from config): {labels}")
        else:
            self.labels = list(
                next(iter(self.datasets.values())).df[self.target].unique()
            )
            self.util.debug(f"Target labels (from database): {labels}")
        glob_conf.set_labels(self.labels)
        self.util.debug(f"loaded databases {dbs}")

    def _import_csv(self, storage):
        # df = pd.read_csv(storage, header=0, index_col=[0,1,2])
        # df.index.set_levels(pd.to_timedelta(df.index.levels[1]), level=1)
        # df.index.set_levels(pd.to_timedelta(df.index.levels[2]), level=2)
        df = audformat.utils.read_csv(storage)
        df.is_labeled = True if self.target in df else False
        # print(df.head())
        return df

    def fill_tests(self):
        """Only fill a new test set"""

        test_dbs = ast.literal_eval(glob_conf.config["DATA"]["tests"])
        self.df_test = pd.DataFrame()
        start_fresh = eval(self.util.config_val("DATA", "no_reuse", "False"))
        store = self.util.get_path("store")
        storage_test = f"{store}extra_testdf.csv"
        if os.path.isfile(storage_test) and not start_fresh:
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
                data.prepare_labels()
                self.df_test = pd.concat(
                    [self.df_test, self.util.make_segmented_index(data.df)]
                )
                self.df_test.is_labeled = data.is_labeled
            self.df_test.got_gender = self.got_gender
            self.df_test.got_speaker = self.got_speaker
            # self.util.set_config_val('FEATS', 'needs_features_extraction', 'True')
            # self.util.set_config_val('FEATS', 'no_reuse', 'True')
            self.df_test["class_labels"] = self.df_test[self.target]
            self.df_test[self.target] = self.label_encoder.transform(
                self.df_test[self.target]
            )
            self.df_test.to_csv(storage_test)

    def fill_train_and_tests(self):
        """Set up train and development sets. The method should be specified in the config."""
        store = self.util.get_path("store")
        storage_test = f"{store}testdf.csv"
        storage_train = f"{store}traindf.csv"
        start_fresh = eval(self.util.config_val("DATA", "no_reuse", "False"))
        if (
            os.path.isfile(storage_train)
            and os.path.isfile(storage_test)
            and not start_fresh
        ):
            self.util.debug(
                f"reusing previously stored {storage_test} and {storage_train}"
            )
            self.df_test = self._import_csv(storage_test)
            # print(f"df_test: {self.df_test}")
            self.df_train = self._import_csv(storage_train)
            # print(f"df_train: {self.df_train}")
        else:
            self.df_train, self.df_test = pd.DataFrame(), pd.DataFrame()
            for d in self.datasets.values():
                d.split()
                d.prepare_labels()
                if d.df_train.shape[0] == 0:
                    self.util.debug(f"warn: {d.name} train empty")
                self.df_train = pd.concat([self.df_train, d.df_train])
                # print(f"df_train: {self.df_train}")
                self.util.copy_flags(d, self.df_train)
                if d.df_test.shape[0] == 0:
                    self.util.debug(f"warn: {d.name} test empty")
                self.df_test = pd.concat([self.df_test, d.df_test])
                self.util.copy_flags(d, self.df_test)
            store = self.util.get_path("store")
            storage_test = f"{store}testdf.csv"
            storage_train = f"{store}traindf.csv"
            self.df_test.to_csv(storage_test)
            self.df_train.to_csv(storage_train)

        self.util.copy_flags(self, self.df_test)
        self.util.copy_flags(self, self.df_train)
        # Try data checks
        datachecker = FileChecker(self.df_train)
        self.df_train = datachecker.all_checks()
        datachecker.set_data(self.df_test)
        self.df_test = datachecker.all_checks()

        # Check for filters
        filter_sample_selection = self.util.config_val(
            "DATA", "filter.sample_selection", "all"
        )
        if filter_sample_selection == "all":
            datafilter = DataFilter(self.df_train)
            self.df_train = datafilter.all_filters()
            datafilter = DataFilter(self.df_test)
            self.df_test = datafilter.all_filters()
        elif filter_sample_selection == "train":
            datafilter = DataFilter(self.df_train)
            self.df_train = datafilter.all_filters()
        elif filter_sample_selection == "test":
            datafilter = DataFilter(self.df_test)
            self.df_test = datafilter.all_filters()
        else:
            self.util.error(
                "unkown filter sample selection specifier"
                f" {filter_sample_selection}, should be [all | train | test]"
            )

        # encode the labels
        if self.util.exp_is_classification():
            datatype = self.util.config_val("DATA", "type", "dummy")
            if datatype == "continuous":
                # if self.df_test.is_labeled:
                #     # remember the target in case they get labelencoded later
                #     self.df_test["class_label"] = self.df_test[self.target]
                test_cats = self.df_test["class_label"].unique()
                # else:
                #     # if there is no target, copy a dummy label
                #     self.df_test = self._add_random_target(self.df_test)
                # if self.df_train.is_labeled:
                #     # remember the target in case they get labelencoded later
                #     self.df_train["class_label"] = self.df_train[self.target]
                train_cats = self.df_train["class_label"].unique()

            else:
                if self.df_test.is_labeled:
                    test_cats = self.df_test[self.target].unique()
                else:
                    # if there is no target, copy a dummy label
                    self.df_test = self._add_random_target(self.df_test).astype("str")
                train_cats = self.df_train[self.target].unique()
                # print(f"df_train: {pd.DataFrame(self.df_train[self.target])}")
                # print(f"train_cats with target {self.target}: {train_cats}")
            if self.df_test.is_labeled:
                if type(test_cats) == np.ndarray:
                    self.util.debug(f"Categories test (nd.array): {test_cats}")
                else:
                    self.util.debug(f"Categories test (list): {list(test_cats)}")
            if type(train_cats) == np.ndarray:
                self.util.debug(f"Categories train (nd.array): {train_cats}")
            else:
                self.util.debug(f"Categories train (list): {list(train_cats)}")

            # encode the labels as numbers
            self.label_encoder = LabelEncoder()
            self.df_train[self.target] = self.label_encoder.fit_transform(
                self.df_train[self.target]
            )
            self.df_test[self.target] = self.label_encoder.transform(
                self.df_test[self.target]
            )
            glob_conf.set_label_encoder(self.label_encoder)
        if self.got_speaker:
            self.util.debug(
                f"{self.df_test.speaker.nunique()} speakers in test and"
                f" {self.df_train.speaker.nunique()} speakers in train"
            )

        target_factor = self.util.config_val("DATA", "target_divide_by", False)
        if target_factor:
            self.df_test[self.target] = self.df_test[self.target] / float(target_factor)
            self.df_train[self.target] = self.df_train[self.target] / float(
                target_factor
            )
            if not self.util.exp_is_classification():
                self.df_test["class_label"] = self.df_test["class_label"] / float(
                    target_factor
                )
                self.df_train["class_label"] = self.df_train["class_label"] / float(
                    target_factor
                )

    def _add_random_target(self, df):
        labels = glob_conf.labels
        a = [None] * len(df)
        for i in range(0, len(df)):
            a[i] = random.choice(labels)
        df[self.target] = a
        return df

    def plot_distribution(self, df_labels):
        """Plot the distribution of samples and speaker per target class and biological sex"""
        plot = Plots()
        sample_selection = self.util.config_val("EXPL", "sample_selection", "all")
        plot.plot_distributions(df_labels)
        if self.got_speaker:
            plot.plot_distributions_speaker(df_labels)

    def extract_test_feats(self):
        self.feats_test = pd.DataFrame()
        feats_name = "_".join(ast.literal_eval(glob_conf.config["DATA"]["tests"]))
        feats_types = self.util.config_val_list("FEATS", "type", ["os"])
        self.feature_extractor = FeatureExtractor(
            self.df_test, feats_types, feats_name, "test"
        )
        self.feats_test = self.feature_extractor.extract()
        self.util.debug(f"Test features shape:{self.feats_test.shape}")

    def extract_feats(self):
        """Extract the features for train and dev sets.

        They will be stored on disk and need to be removed manually.

        The string FEATS.feats_type is read from the config, defaults to os.

        """
        df_train, df_test = self.df_train, self.df_test
        feats_name = "_".join(ast.literal_eval(glob_conf.config["DATA"]["databases"]))
        self.feats_test, self.feats_train = pd.DataFrame(), pd.DataFrame()
        feats_types = self.util.config_val_list("FEATS", "type", ["os"])
        self.feature_extractor = FeatureExtractor(
            df_train, feats_types, feats_name, "train"
        )
        self.feats_train = self.feature_extractor.extract()
        self.feature_extractor = FeatureExtractor(
            df_test, feats_types, feats_name, "test"
        )
        self.feats_test = self.feature_extractor.extract()
        self.util.debug(
            f"All features: train shape : {self.feats_train.shape}, test"
            f" shape:{self.feats_test.shape}"
        )
        if self.feats_train.shape[0] < self.df_train.shape[0]:
            self.util.warn(
                f"train feats ({self.feats_train.shape[0]}) != train labels"
                f" ({self.df_train.shape[0]})"
            )
            self.df_train = self.df_train[
                self.df_train.index.isin(self.feats_train.index)
            ]
            self.util.warn(f"new train labels shape: {self.df_train.shape[0]}")
        if self.feats_test.shape[0] < self.df_test.shape[0]:
            self.util.warn(
                f"test feats ({self.feats_test.shape[0]}) != test labels"
                f" ({self.df_test.shape[0]})"
            )
            self.df_test = self.df_test[self.df_test.index.isin(self.feats_test.index)]
            self.util.warn(f"mew test labels shape: {self.df_test.shape[0]}")

        self._check_scale()

    def augment(self):
        """
        Augment the selected samples
        """
        from nkululeko.augmenting.augmenter import Augmenter

        sample_selection = self.util.config_val("AUGMENT", "sample_selection", "all")
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

        augmenter = Augmenter(df)
        df_ret = augmenter.augment(sample_selection)
        return df_ret

    def autopredict(self):
        """
        Predict labels for samples with existing models and add to the dataframe.
        """
        sample_selection = self.util.config_val("PREDICT", "split", "all")
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
        targets = self.util.config_val_list("PREDICT", "targets", ["gender"])
        for target in targets:
            if target == "gender":
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
            else:
                self.util.error(f"unknown auto predict target: {target}")
        return df

    def random_splice(self):
        """
        Random-splice the selected samples
        """
        from nkululeko.augmenting.randomsplicer import Randomsplicer

        sample_selection = self.util.config_val("AUGMENT", "sample_selection", "all")
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
        """
        Do a feature exploration

        """

        plot_feats = eval(
            self.util.config_val("EXPL", "feature_distributions", "False")
        )
        sample_selection = self.util.config_val("EXPL", "sample_selection", "all")
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

        if self.util.config_val("EXPL", "value_counts", False):
            self.plot_distribution(df_labels)

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

        if plot_feats:
            feat_analyser = FeatureAnalyser(sample_selection, df_labels, df_feats)
            feat_analyser.analyse()

        # check if a scatterplot should be done
        scatter_var = eval(self.util.config_val("EXPL", "scatter", "False"))
        scatter_target = self.util.config_val(
            "EXPL", "scatter.target", "['class_label']"
        )
        if scatter_var:
            scatters = ast.literal_eval(glob_conf.config["EXPL"]["scatter"])
            scat_targets = ast.literal_eval(scatter_target)
            plots = Plots()
            for scat_target in scat_targets:
                if self.util.is_categorical(df_labels[scat_target]):
                    for scatter in scatters:
                        plots.scatter_plot(df_feats, df_labels, scat_target, scatter)
                else:
                    self.util.debug(
                        f"{self.name}: binning continuous variable to categories"
                    )
                    cat_vals = self.util.continuous_to_categorical(
                        df_labels[scat_target]
                    )
                    df_labels[f"{scat_target}_bins"] = cat_vals.values
                    for scatter in scatters:
                        plots.scatter_plot(
                            df_feats, df_labels, f"{scat_target}_bins", scatter
                        )

    def _check_scale(self):
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
            )
            self.feats_train, self.feats_test = self.scaler_feats.scale()
            # store versions
            self.util.save_to_store(self.feats_train, "feats_train_scaled")
            self.util.save_to_store(self.feats_test, "feats_test_scaled")

    def init_runmanager(self):
        """Initialize the manager object for the runs."""
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
        save = self.util.config_val("EXP", "save", False)
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
                "plot combined speaker predictions not possible for cross" " validation"
            )
            return
        best = self.get_best_report(self.reports)
        # if not best.is_classification:
        #     best.continuous_to_categorical()
        truths = best.truths
        preds = best.preds
        speakers = self.df_test.speaker.values
        print(f"{len(truths)} {len(preds)} {len(speakers) }")
        df = pd.DataFrame(data={"truth": truths, "pred": preds, "speaker": speakers})
        plot_name = "result_combined_per_speaker"
        self.util.debug(
            f"plotting speaker combination ({function}) confusion matrix to"
            f" {plot_name}"
        )
        best.plot_per_speaker(df, plot_name, function)

    def get_best_report(self, reports):
        return self.runmgr.get_best_result(reports)

    def print_best_model(self):
        self.runmgr.print_best_result_runs()

    def demo(self, file, is_list, outfile):
        model = self.runmgr.get_best_model()
        labelEncoder = None
        try:
            labelEncoder = self.label_encoder
        except AttributeError:
            pass
        demo = Demo_predictor(
            model, file, is_list, self.feature_extractor, labelEncoder, outfile
        )
        demo.run_demo()

    def predict_test_and_save(self, result_name):
        model = self.runmgr.get_best_model()
        model.set_testdata(self.df_test, self.feats_test)
        test_predictor = Test_predictor(
            model, self.df_test, self.label_encoder, result_name
        )
        test_predictor.predict_and_store()

    def load(self, filename):
        f = open(filename, "rb")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
        glob_conf.set_labels(self.labels)

    def save(self, filename):
        if self.runmgr.modelrunner.model.is_ANN():
            self.runmgr.modelrunner.model = None
            self.util.warn(
                f"Save experiment: Can't pickle the learning model so saving without it."
            )
        try:
            f = open(filename, "wb")
            pickle.dump(self.__dict__, f)
            f.close()
        except TypeError:
            self.feature_extractor.featExtractor.model = None
            f = open(filename, "wb")
            pickle.dump(self.__dict__, f)
            f.close()
            self.util.warn(
                f"Save experiment: Can't pickle the feature extraction model so saving without it."
            )
        except (AttributeError, RuntimeError) as error:
            self.util.warn(f"Save experiment: Can't pickle local object: {error}")

    def save_onnx(self, filename):
        # export the model to onnx
        model = self.runmgr.get_best_model()
        if model.is_ANN():
            print("converting to onnx from torch")
        else:
            from skl2onnx import to_onnx

            print("converting to onnx from sklearn")
        # save the rest
        f = open(filename, "wb")
        pickle.dump(self.__dict__, f)
        f.close()
