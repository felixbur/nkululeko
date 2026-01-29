# modelrunner.py

import ast
import pandas as pd

from nkululeko import glob_conf
from nkululeko.utils.util import Util
from nkululeko.balance import DataBalancer


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
        # Initialize best_performance based on metric direction
        if self.util.high_is_good():
            self.best_performance = 0
        else:
            self.best_performance = 100000
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
            epoch = epoch_num
            report.set_id(self.run, epoch)
            plot_name = self.util.get_plot_name() + f"_{self.run}_{epoch:03d}_cnf"
            reports.append(report)
            test_score_metric = reports[-1].get_result().get_test_result()
            metric_label = self.util.config_val("MODEL", "measure", "uar").upper()
            performance = float(test_score_metric.split(" ")[1])
            formatted_performance = f"{performance:.4f}"
            self.util.debug(
                f"run: {self.run} epoch: {epoch}: result: "
                f"{test_score_metric.split(' ')[0]} {formatted_performance} {metric_label}"
            )
            if plot_epochs:
                self.util.debug(f"plotting conf matrix to {plot_name}")
                report.plot_confmatrix(plot_name, epoch)
        else:
            # for all epochs
            for epoch_index, epoch in enumerate(range(epoch_num)):
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
                # Extract performance value and format to 4 digits with leading zeros
                performance = float(test_score_metric.split(" ")[1])
                formatted_performance = f"{performance:.4f}"
                metric_label = self.util.config_val("MODEL", "measure", "uar").upper()
                self.util.debug(
                    f"run: {self.run} epoch: {epoch}: result: {test_score_metric.split(' ')[0]} {formatted_performance} {metric_label}"
                )
                # print(f"performance: {performance.split(' ')[1]}")
                # Update best performance based on metric direction (lower is better for EER, higher for UAR/ACC)
                if self.util.high_is_good():
                    if performance > self.best_performance:
                        self.best_performance = performance
                        self.best_epoch = epoch
                else:
                    if performance < self.best_performance:
                        self.best_performance = performance
                        self.best_epoch = epoch
                if plot_epochs:
                    self.util.debug(f"plotting conf matrix to {plot_name}")
                    report.plot_confmatrix(plot_name, epoch)

                # check if we need should not store the model
                save_models = ast.literal_eval(
                    self.util.config_val("MODEL", "save", "True")
                )
                if save_models:  # in any case the model needs to be stored to disk.
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
        return reports, epoch

    def eval_last_model(self, df_test, feats_test):
        self.model.reset_test(df_test, feats_test)
        report = self.model.predict()
        report.set_id(self.run, 0)
        return report

    def eval_specific_model(self, model, df_test, feats_test):
        self.model = model
        self.util.debug(f"evaluating model: {self.model.store_path}")
        self.model.reset_test(df_test, feats_test)
        report = self.model.predict()
        report.set_id(self.run, 0)
        return report

    def _check_balancing(self):
        if self.util.config_val("EXP", "balancing", False):
            self.util.debug("balancing data")
            self.df_train, self.df_test = self.util.balance_data(
                self.df_train, self.df_test
            )
            self.util.debug(f"new train size: {self.df_train.shape}")
            self.util.debug(f"new test size: {self.df_test.shape}")

    def _select_model(self, model_type):
        self._check_balancing()
        self._check_feature_balancing()

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
        elif model_type == "adm":
            from nkululeko.models.model_adm import ADMModel

            self.model = ADMModel(
                self.df_train, self.df_test, self.feats_train, self.feats_test
            )
        elif model_type == "adad":
            from nkululeko.models.model_adad import ADADModel

            self.model = ADADModel(
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

    def _check_feature_balancing(self):
        """Check and apply feature balancing using the dedicated DataBalancer class."""
        balancing = self.util.config_val("FEATS", "balancing", False)
        if balancing:
            self.util.debug("Applying feature balancing using DataBalancer")

            # Get random state from config, fallback to 42 for backward compatibility
            random_state = int(
                self.util.config_val("FEATS", "balancing_random_state", 42)
            )

            # Initialize the data balancer with configurable random state
            balancer = DataBalancer(random_state=random_state)

            # Apply balancing
            self.df_train, self.feats_train = balancer.balance_features(
                df_train=self.df_train,
                feats_train=self.feats_train,
                target_column=self.target,
                method=balancing,
            )
