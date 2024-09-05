# modelrunner.py

import pandas as pd

from nkululeko import glob_conf
from nkululeko.utils.util import Util


class Modelrunner:
    """Class to model one run."""

    def __init__(self, df_train, df_test, feats_train, feats_test, run):
        """Constructor setting up the dataframes.

        Args:
            df_train: train dataframe
            df_test: test dataframe
            feats_train: train features
            feats_train: test features
        """
        self.df_train, self.df_test, self.feats_train, self.feats_test = (
            df_train,
            df_test,
            feats_train,
            feats_test,
        )
        self.util = Util("modelrunner")
        self.run = run
        self.target = glob_conf.config["DATA"]["target"]
        # intialize a new model
        model_type = glob_conf.config["MODEL"]["type"]
        self._select_model(model_type)
        self.best_performance = 0
        self.best_epoch = 0

    def do_epochs(self):
        # initialze results
        reports = []
        plot_epochs = self.util.config_val("PLOT", "epochs", False)
        only_test = self.util.config_val("MODEL", "only_test", False)
        epoch_num = int(self.util.config_val("EXP", "epochs", 1))
        if not self.model.is_ann() and epoch_num > 1:
            self.util.warn(f"setting epoch num to 1 (was {epoch_num}) if model not ANN")
            epoch_num = 1
            glob_conf.config["EXP"]["epochs"] = "1"
        patience = self.util.config_val("MODEL", "patience", False)
        patience_counter = -1
        if self.util.high_is_good():
            highest = 0
        else:
            highest = 100000
        if self.model.model_type == "finetuned":
            # epochs are handled by Huggingface API
            self.model.train()
            report = self.model.predict()
            # todo: findout the best epoch -> no need
            # since load_best_model_at_end is given in training args
            epoch = epoch_num
            report.set_id(self.run, epoch)
            plot_name = self.util.get_plot_name() + f"_{self.run}_{epoch:03d}_cnf"
            reports.append(report)
            self.util.debug(
                f"run: {self.run} epoch: {epoch}: result: "
                f"{reports[-1].get_result().get_test_result()}"
            )
            if plot_epochs:
                self.util.debug(f"plotting conf matrix to {plot_name}")
                report.plot_confmatrix(plot_name, epoch)
        else:
            # for all epochs
            for epoch in range(epoch_num):
                if only_test:
                    self.model.load(self.run, epoch)
                    self.util.debug(f"reusing model: {self.model.store_path}")
                    self.model.reset_test(self.df_test, self.feats_test)
                else:
                    self.model.set_id(self.run, epoch)
                    self.model.train()
                report = self.model.predict()
                report.set_id(self.run, epoch)
                plot_name = self.util.get_plot_name() + f"_{self.run}_{epoch:03d}_cnf"
                reports.append(report)
                test_score_metric = report.get_result().get_test_result()
                self.util.debug(
                    f"run: {self.run} epoch: {epoch}: result: {test_score_metric}"
                )
                # print(f"performance: {performance.split(' ')[1]}")
                performance = float(test_score_metric.split(" ")[1])
                if performance > self.best_performance:
                    self.best_performance = performance
                    self.best_epoch = epoch
                if plot_epochs:
                    self.util.debug(f"plotting conf matrix to {plot_name}")
                    report.plot_confmatrix(plot_name, epoch)
                store_models = self.util.config_val("EXP", "save", False)
                plot_best_model = self.util.config_val("PLOT", "best_model", False)
                if (store_models or plot_best_model) and (
                    not only_test
                ):  # in any case the model needs to be stored to disk.
                    self.model.store()
                if patience:
                    patience = int(patience)
                    result = report.result.get_result()
                    if self.util.high_is_good():
                        if result > highest:
                            highest = result
                            patience_counter = 0
                        else:
                            patience_counter += 1
                    else:
                        if result < highest:
                            highest = result
                            patience_counter = 0
                        else:
                            patience_counter += 1
                    if patience_counter >= patience:
                        self.util.debug(
                            f"reached patience ({str(patience)}): early stopping"
                        )
                        break
        # After training, report the best performance and epoch
        best_report = reports[self.best_epoch]
        # self.util.debug(f"Best score at epoch: {self.best_epoch}, UAR: {self.best_performance}") # move to reporter below

        if not plot_epochs:
            # Do at least one confusion matrix plot
            self.util.debug(f"plotting confusion matrix to {plot_name}")
            best_report.plot_confmatrix(plot_name, self.best_epoch)
        return reports, epoch

    def _select_model(self, model_type):
        self._check_balancing()

        if model_type == "svm":
            from nkululeko.models.model_svm import SVM_model

            self.model = SVM_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "svr":
            from nkululeko.models.model_svr import SVR_model

            self.model = SVR_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "xgb":
            from nkululeko.models.model_xgb import XGB_model

            self.model = XGB_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "xgr":
            from nkululeko.models.model_xgr import XGR_model

            self.model = XGR_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "bayes":
            from nkululeko.models.model_bayes import Bayes_model

            self.model = Bayes_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "finetune":
            from nkululeko.models.model_tuned import TunedModel

            self.model = TunedModel(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "gmm":
            from nkululeko.models.model_gmm import GMM_model

            self.model = GMM_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "knn":
            from nkululeko.models.model_knn import KNN_model

            self.model = KNN_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "knn_reg":
            from nkululeko.models.model_knn_reg import KNN_reg_model

            self.model = KNN_reg_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "lin_reg":
            from nkululeko.models.model_lin_reg import Lin_reg_model

            self.model = Lin_reg_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "tree":
            from nkululeko.models.model_tree import Tree_model

            self.model = Tree_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "tree_reg":
            from nkululeko.models.model_tree_reg import Tree_reg_model

            self.model = Tree_reg_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "cnn":
            from nkululeko.models.model_cnn import CNNModel

            self.model = CNNModel(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "mlp":
            from nkululeko.models.model_mlp import MLPModel

            self.model = MLPModel(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "mlp_reg":
            from nkululeko.models.model_mlp_regression import MLP_Reg_model

            self.model = MLP_Reg_model(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        else:
            self.util.error(f"unknown model type: '{model_type}'")
        if self.util.exp_is_classification() and not self.model.is_classifier:
            self.util.error(
                "Experiment type set to classification but model type is not a"
                " classifier"
            )
        return self.model

    def _check_balancing(self):
        balancing = self.util.config_val("FEATS", "balancing", False)
        if balancing:
            orig_size = self.feats_train.shape[0]
            self.util.debug(f"balancing the training features with: {balancing}")
            if balancing == "ros":
                from imblearn.over_sampling import RandomOverSampler

                sampler = RandomOverSampler(random_state=42)
                X_res, y_res = sampler.fit_resample(
                    self.feats_train, self.df_train[self.target]
                )
            elif balancing == "smote":
                from imblearn.over_sampling import SMOTE

                sampler = SMOTE(random_state=42)
                X_res, y_res = sampler.fit_resample(
                    self.feats_train, self.df_train[self.target]
                )
            elif balancing == "adasyn":
                from imblearn.over_sampling import ADASYN

                sampler = ADASYN(random_state=42)
                X_res, y_res = sampler.fit_resample(
                    self.feats_train, self.df_train[self.target]
                )
            else:
                self.util.error(
                    f"unknown balancing algorithm: {balancing} (should be [ros|smote|adasyn])"
                )

            self.feats_train = X_res
            self.df_train = pd.DataFrame({self.target: y_res}, index=X_res.index)
            self.util.debug(
                f"balanced with: {balancing}, new size: {X_res.shape[0]} (was {orig_size})"
            )
            le = glob_conf.label_encoder
            res = y_res.value_counts()
            resd = {}
            for i, e in enumerate(le.inverse_transform(res.index.values)):
                resd[e] = res.values[i]
            self.util.debug(f"{resd})")
