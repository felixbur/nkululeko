# model.py
import ast
import pickle
import random

import numpy as np
import pandas as pd
import sklearn.utils
from joblib import parallel_backend
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold

import nkululeko.glob_conf as glob_conf
from nkululeko.reporting.reporter import Reporter
from nkululeko.utils.util import Util


class Model:
    """Generic model class for linear (non-neural) algorithms."""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        """Constructor taking the configuration and all dataframes."""
        self.name = "undefined"
        self.df_train, self.df_test, self.feats_train, self.feats_test = (
            df_train,
            df_test,
            feats_train,
            feats_test,
        )
        self.model_type = "classic"
        self.util = Util("model")
        self.target = self.util.config_val("DATA", "target", "emotion")
        self.run = 0
        self.epoch = 0
        self.logo = self.util.config_val("MODEL", "logo", False)
        self.xfoldx = self.util.config_val("MODEL", "k_fold_cross", False)
        self.n_jobs = int(self.util.config_val("MODEL", "n_jobs", "8"))

    def set_model_type(self, type):
        self.model_type = type

    def is_ann(self):
        if (self.model_type == "ann") or (self.model_type == "finetuned"):
            return True
        else:
            return False

    def set_testdata(self, data_df, feats_df):
        self.df_test, self.feats_test = data_df, feats_df

    def reset_test(self, df_test, feats_test):
        self.df_test, self.feats_test = df_test, feats_test

    def set_id(self, run, epoch):
        self.run = run
        self.epoch = epoch
        dir = self.util.get_path("model_dir")
        name = f"{self.util.get_exp_name(only_train=True)}_{self.run}_{self.epoch:03d}.model"
        self.store_path = dir + name

    def _x_fold_cross(self):
        # ignore train and test sets and do a x-fold-cross  evaluation
        self.util.debug(
            f"ignoring splits and doing {self.xfoldx} fold cross validation"
        )
        feats = pd.concat([self.feats_train, self.feats_test])
        annos = pd.concat([self.df_train, self.df_test])
        targets = annos[self.target]
        _skf = StratifiedKFold(n_splits=int(self.xfoldx))
        truths, preds, results = [], [], []
        g_index = 0
        # leave-one-speaker loop
        for train_index, test_index in _skf.split(
            feats,
            targets,
        ):
            train_x = feats.iloc[train_index].to_numpy()
            train_y = targets[train_index]
            with parallel_backend("threading", n_jobs=self.n_jobs):
                self.clf.fit(train_x, train_y)
            truth_x = feats.iloc[test_index].to_numpy()
            truth_y = targets[test_index]
            predict_y = self.clf.predict(truth_x)
            report = Reporter(truth_y.astype(float), predict_y, self.run, self.epoch)
            self.util.debug(
                f"result for fold {g_index}:"
                f" {report.get_result().get_test_result()} "
            )
            results.append(float(report.get_result().test))
            truths.append(truth_y)
            preds.append(predict_y)
            g_index += 1

        # combine speaker folds
        truth = pd.concat(truths)
        truth.name = "truth"
        pred = pd.Series(
            np.concatenate(preds),
            index=truth.index,
            name="prediction",
        )
        self.truths = truth
        self.preds = pred
        results = np.asarray(results)
        self.util.debug(
            f"KFOLD: {self.xfoldx} folds: mean {results.mean():.3f}, std:"
            f" {results.std():.3f}"
        )

    def _do_logo(self):
        # ignore train and test sets and do a "leave one speaker group out"  evaluation
        logo = int(self.logo)
        feats = pd.concat([self.feats_train, self.feats_test])
        annos = pd.concat([self.df_train, self.df_test])
        targets = annos[self.target]
        _logo = LeaveOneGroupOut()
        truths, preds, results = [], [], []
        # get unique list of speakers
        speakers = annos["speaker"].unique()
        # check for folds columns
        if "fold" not in annos.columns:
            self.util.debug(f"creating random folds for {logo} groups")
            # create a random dictionary of groups
            sdict = {}
            # randomize the speaker order
            random.shuffle(speakers)
            folds = list(range(logo))
            for i, s in enumerate(speakers):
                sdict[s] = folds[i % len(folds)]
            # add this to the annotations
            annos["fold"] = annos["speaker"].apply(lambda x: str(sdict[x]))
            fold_count = self.logo
        else:
            fold_count = annos["fold"].nunique()
            self.util.debug(f"using existing folds for {fold_count} groups")
        g_index = 0
        self.util.debug(f"ignoring splits and doing LOGO with {fold_count} groups")
        # leave-one-group loop
        for train_index, test_index in _logo.split(
            feats,
            targets,
            groups=annos["fold"],
        ):
            train_x = feats.iloc[train_index].to_numpy()
            train_y = targets.iloc[train_index]
            with parallel_backend("threading", n_jobs=self.n_jobs):
                self.clf.fit(train_x, train_y)

            truth_x = feats.iloc[test_index].to_numpy()
            truth_y = targets.iloc[test_index]
            predict_y = self.clf.predict(truth_x)
            report = Reporter(truth_y.astype(float), predict_y, self.run, self.epoch)
            result = report.get_result().get_test_result()
            self.util.debug(f"result for speaker group {g_index}: {result} ")
            results.append(float(report.get_result().test))
            truths.append(truth_y)
            preds.append(predict_y)
            g_index += 1

        # combine speaker folds
        truth = pd.concat(truths)
        truth.name = "truth"
        pred = pd.Series(
            np.concatenate(preds),
            index=truth.index,
            name="prediction",
        )
        self.truths = truth
        self.preds = pred
        results = np.asarray(results)
        self.util.debug(
            f"LOGO: {self.logo} folds: mean {results.mean():.3f}, std:"
            f" {results.std():.3f}"
        )

    def train(self):
        """Train the model."""
        # # first check if the model already has been trained
        # if os.path.isfile(self.store_path):
        #     self.load(self.run, self.epoch)
        #     self.util.debug(f'reusing model: {self.store_path}')
        #     return

        # then if leave one speaker group out validation is wanted
        if self.logo:
            self._do_logo()
            return
        # then if x fold cross validation is wanted
        if self.xfoldx:
            self._x_fold_cross()
            return

        # check for NANs in the features
        # set up the data_loaders
        if self.feats_train.isna().to_numpy().any():
            self.util.debug(
                "Model, train: replacing"
                f" {self.feats_train.isna().sum().sum()} NANs with 0"
            )
            self.feats_train = self.feats_train.fillna(0)
        # remove labels from features
        feats = self.feats_train.to_numpy()
        # compute class weights
        if self.util.config_val("MODEL", "class_weight", False):
            self.classes_weights = sklearn.utils.class_weight.compute_sample_weight(
                class_weight="balanced", y=self.df_train[self.target]
            )

        tuning_params = self.util.config_val("MODEL", "tuning_params", False)
        with parallel_backend("threading", n_jobs=self.n_jobs):
            if tuning_params:
                # tune the model meta parameters
                tuning_params = ast.literal_eval(tuning_params)
                tuned_params = {}
                try:
                    scoring = glob_conf.config["MODEL"]["scoring"]
                except KeyError:
                    self.util.error("got tuning params but no scoring")
                for param in tuning_params:
                    values = ast.literal_eval(glob_conf.config["MODEL"][param])
                    tuned_params[param] = values
                self.util.debug(f"tuning on {tuned_params}")
                self.clf = GridSearchCV(
                    self.clf, tuned_params, refit=True, verbose=3, scoring=scoring
                )
                try:
                    class_weight = eval(
                        self.util.config_val("MODEL", "class_weight", "False")
                    )
                    if class_weight:
                        self.util.debug("using class weight")
                        self.clf.fit(
                            feats,
                            self.df_train[self.target],
                            sample_weight=self.classes_weights,
                        )
                    else:
                        self.clf.fit(feats, self.df_train[self.target])
                except KeyError:
                    self.clf.fit(feats, self.df_train[self.target])
                self.util.debug(f"winner parameters: {self.clf.best_params_}")
            else:
                class_weight = self.util.config_val("MODEL", "class_weight", False)
                if class_weight:
                    self.util.debug("using class weight")
                    self.clf.fit(
                        feats,
                        self.df_train[self.target],
                        sample_weight=self.classes_weights,
                    )
                else:
                    labels = self.df_train[self.target]
                    self.clf.fit(feats, labels)

    def get_predictions(self):
        #        predictions = self.clf.predict(self.feats_test.to_numpy())
        if self.util.exp_is_classification():
            # make a dataframe for the class probabilities
            proba_d = {}
            for c in self.clf.classes_:
                proba_d[c] = []
            # get the class probabilities
            predictions = self.clf.predict_proba(self.feats_test.to_numpy())
            # pred = self.clf.predict(features)
            for i, c in enumerate(self.clf.classes_):
                proba_d[c] = list(predictions.T[i])
            probas = pd.DataFrame(proba_d)
            probas = probas.set_index(self.feats_test.index)
            predictions = probas.idxmax(axis=1).values
        else:
            predictions = self.clf.predict(self.feats_test.to_numpy())
            probas = None

        return predictions, probas

    def predict(self):
        if self.feats_test.isna().to_numpy().any():
            self.util.debug(
                "Model, test: replacing"
                f" {self.feats_test.isna().sum().sum()} NANs with 0"
            )
            self.feats_test = self.feats_test.fillna(0)
        if self.logo or self.xfoldx:
            report = Reporter(
                self.truths.astype(float), self.preds, self.run, self.epoch
            )
            return report
        """Predict the whole eval feature set"""
        predictions, probas = self.get_predictions()

        report = Reporter(
            self.df_test[self.target].to_numpy().astype(float),
            predictions,
            self.run,
            self.epoch,
            probas=probas,
        )
        report.print_probabilities()
        return report

    def get_type(self):
        return "generic"

    def predict_sample(self, features):
        """Predict one sample"""
        prediction = {}
        if self.util.exp_is_classification():
            # get the class probabilities
            predictions = self.clf.predict_proba(features)
            # pred = self.clf.predict(features)
            for i in range(len(self.clf.classes_)):
                cat = self.clf.classes_[i]
                prediction[cat] = predictions[0][i]
        else:
            predictions = self.clf.predict(features)
            prediction = predictions[0]
        return prediction

    def store(self):
        with open(self.store_path, "wb") as handle:
            pickle.dump(self.clf, handle)

    def load(self, run, epoch):
        self.set_id(run, epoch)
        dir = self.util.get_path("model_dir")
        name = f"{self.util.get_exp_name(only_train=True)}_{self.run}_{self.epoch:03d}.model"
        try:
            with open(dir + name, "rb") as handle:
                self.clf = pickle.load(handle)
        except FileNotFoundError as fe:
            self.util.error(
                f"Did you forget to store your models? needs: \n[MODEL]\nsave=True\n{fe}"
            )

    def load_path(self, path, run, epoch):
        self.set_id(run, epoch)
        with open(path, "rb") as handle:
            self.clf = pickle.load(handle)
