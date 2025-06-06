# model_xgb.py

import os
from xgboost import XGBClassifier

import nkululeko.glob_conf as glob_conf
from nkululeko.models.model import Model


class XGB_model(Model):
    """An XGBoost model with early stopping support"""

    def __init__(self, df_train, df_test, feats_train, feats_test):
        super().__init__(df_train, df_test, feats_train, feats_test)
        self.name = "xgb"
        self.is_classifier = True

        # Configure XGBoost parameters
        xgb_params = {}

        # Get early stopping configuration
        self.early_stopping_rounds = self.util.config_val(
            "MODEL", "early_stopping_rounds", False
        )
        self.eval_metric = self.util.config_val("MODEL", "eval_metric", "logloss")

        # Set up other XGBoost parameters that can be configured
        n_estimators = self.util.config_val("MODEL", "n_estimators", 100)
        max_depth = self.util.config_val("MODEL", "max_depth", 6)
        learning_rate = self.util.config_val("MODEL", "learning_rate", 0.3)
        subsample = self.util.config_val("MODEL", "subsample", 1.0)

        xgb_params["n_estimators"] = int(n_estimators)
        xgb_params["max_depth"] = int(max_depth)
        xgb_params["learning_rate"] = float(learning_rate)
        xgb_params["subsample"] = float(subsample)

        # Set random state for reproducibility
        xgb_params["random_state"] = 42

        # Add early stopping parameters to model initialization if configured
        if self.early_stopping_rounds:
            xgb_params["early_stopping_rounds"] = int(self.early_stopping_rounds)
            xgb_params["eval_metric"] = self.eval_metric

        # Initialize classifier with parameters
        self.clf = XGBClassifier(**xgb_params)

    def train(self):
        """Train the XGBoost model with optional early stopping."""
        # Check if NANs in features and handle them
        if self.feats_train.isna().to_numpy().any():
            self.util.debug(
                "Model, train: replacing"
                f" {self.feats_train.isna().sum().sum()} NANs with 0"
            )
            self.feats_train = self.feats_train.fillna(0)

        feats = self.feats_train.to_numpy()
        labels = self.df_train[self.target]

        # Configure fitting parameters
        fit_params = {}

        # Check if early stopping is configured
        if self.early_stopping_rounds:
            # Check if we're in split3 mode (train/dev/test) where validation data is available
            import ast

            split3 = ast.literal_eval(
                self.util.config_val("EXP", "traindevtest", "False")
            )

            if split3 and self.feats_test is not None and self.df_test is not None:
                # In split3 mode, self.feats_test and self.df_test are actually the dev set
                feats_dev = self.feats_test.to_numpy()
                labels_dev = self.df_test[self.target]

                # Handle NANs in dev features
                if self.feats_test.isna().to_numpy().any():
                    self.util.debug(
                        "Model, dev: replacing"
                        f" {self.feats_test.isna().sum().sum()} NANs with 0"
                    )
                    feats_dev = self.feats_test.fillna(0).to_numpy()

                # Set up early stopping with validation data
                eval_set = [(feats, labels), (feats_dev, labels_dev)]
                fit_params["eval_set"] = eval_set
                fit_params["verbose"] = True

                self.util.debug(
                    f"Training XGBoost with early stopping (using dev set):"
                )
                self.util.debug(
                    f"  - early_stopping_rounds: {self.early_stopping_rounds}"
                )
                self.util.debug(f"  - eval_metric: {self.eval_metric}")
                self.util.debug(f"  - validation set size: {feats_dev.shape[0]}")
            else:
                # For train/test split only: use a portion of training data for validation
                from sklearn.model_selection import train_test_split

                # Get validation split ratio (default 0.2 = 20% of training data)
                val_split = float(
                    self.util.config_val("MODEL", "validation_split", 0.2)
                )

                # Split training data into train and validation
                feats_train_split, feats_val, labels_train_split, labels_val = (
                    train_test_split(
                        feats,
                        labels,
                        test_size=val_split,
                        random_state=42,
                        stratify=labels,
                    )
                )

                # Set up early stopping with validation split
                eval_set = [
                    (feats_train_split, labels_train_split),
                    (feats_val, labels_val),
                ]
                fit_params["eval_set"] = eval_set
                fit_params["verbose"] = True

                # Use the split training data for actual training
                feats = feats_train_split
                labels = labels_train_split

                self.util.debug(
                    f"Training XGBoost with early stopping (using validation split):"
                )
                self.util.debug(
                    f"  - early_stopping_rounds: {self.early_stopping_rounds}"
                )
                self.util.debug(f"  - eval_metric: {self.eval_metric}")
                self.util.debug(f"  - validation_split: {val_split}")
                self.util.debug(f"  - training set size: {feats_train_split.shape[0]}")
                self.util.debug(f"  - validation set size: {feats_val.shape[0]}")

        # Handle class weights if configured
        class_weight = self.util.config_val("MODEL", "class_weight", False)
        if class_weight:
            import sklearn.utils.class_weight

            self.util.debug("using class weight")
            classes_weights = sklearn.utils.class_weight.compute_sample_weight(
                class_weight="balanced", y=labels
            )
            fit_params["sample_weight"] = classes_weights

        # Train the model
        self.clf.fit(feats, labels, **fit_params)

        # Log information about the trained model
        if hasattr(self.clf, "best_iteration"):
            self.util.debug(f"Best iteration: {self.clf.best_iteration}")
        if hasattr(self.clf, "best_score"):
            self.util.debug(f"Best score: {self.clf.best_score}")

    def get_type(self):
        return "xgb"
