#!/usr/bin/env python3

import argparse
import ast
import configparser
import itertools
import os
import random
import sys
import time

import numpy as np

from nkululeko.constants import VERSION
from nkululeko.utils.util import Util


class OptimizationRunner:
    """Hyperparameter optimization runner for nkululeko experiments."""

    def __init__(self, config):
        self.config = config
        self.util = Util("optim")
        self.results = []
        self.model_type = None  # Will be set when parsing OPTIM params
        # New: Optimization strategy configuration
        self.search_strategy = "grid"  # Default values
        self.n_iter = 50
        self.cv_folds = 3
        self.metric = "accuracy"
        self.random_state = 42  # Default random state for reproducibility

    def parse_optim_params(self):
        """Parse OPTIM section parameters into search spaces."""
        if "OPTIM" not in self.config:
            self.util.error("No [OPTIM] section found in configuration")

        optim_config = self.config["OPTIM"]
        self.model_type = optim_config.get("model", "mlp")

        # Parse optimization strategy settings
        self.search_strategy = optim_config.get("search_strategy", "grid")
        self.n_iter = int(optim_config.get("n_iter", "50"))
        self.cv_folds = int(optim_config.get("cv_folds", "3"))
        self.random_state = int(optim_config.get("random_state", "42"))
        
        # Set global random seeds for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
        self.util.debug(f"Using random state: {self.random_state} for reproducibility")
        
        self.metric = optim_config.get("metric", "accuracy").lower()

        self.util.debug(f"Parsed metric from config: '{self.metric}'")  # Debug line

        param_specs = {}
        for key, value in optim_config.items():
            if key in ["model", "search_strategy", "n_iter", "cv_folds", "metric", "random_state"]:
                continue
            param_specs[key] = self._parse_param_spec(key, value)

        return param_specs

    def _parse_param_spec(self, param_name, param_value):
        """Parse individual parameter specification."""
        try:
            parsed = ast.literal_eval(param_value)
        except (ValueError, SyntaxError) as e:
            self.util.debug(
                f"Could not parse parameter {param_name}={param_value} as literal, treating as string: {e}"
            )
            if isinstance(param_value, str):
                return [param_value]
            return param_value

        # Check for inefficient learning rate ranges and suggest better alternatives
        if param_name == "lr" and isinstance(parsed, tuple) and len(parsed) == 3:
            min_val, max_val, step = parsed
            if step <= 0.0001 and (max_val - min_val) / step > 20:
                self.util.debug(
                    f"WARNING: Learning rate range {param_value} will generate {int((max_val - min_val) / step)} values!"
                )
                self.util.debug(
                    "Consider using discrete values like [0.0001, 0.001, 0.01, 0.1] or range (0.0001, 0.1) for log-scale sampling"
                )

        if isinstance(parsed, tuple):
            if len(parsed) == 2:
                return self._generate_range(parsed[0], parsed[1], param_name)
            elif len(parsed) == 3:
                return self._generate_range_with_step(
                    parsed[0], parsed[1], parsed[2], param_name
                )
            else:
                self.util.error(
                    f"Invalid tuple format for parameter {param_name}: {param_value}. Expected (min, max) or (min, max, step)"
                )
                return [parsed[0]]  # Fallback to first value
        elif isinstance(parsed, list):
            return parsed
        else:
            return [parsed]

    def _generate_range(self, min_val, max_val, param_name):
        """Generate parameter range based on parameter type."""
        if param_name in ["nlayers"]:
            return list(range(min_val, max_val + 1))
        elif param_name in ["nnodes", "bs"]:
            result = []
            current = min_val
            while current <= max_val:
                result.append(current)
                current *= 2
            return result
        elif param_name in ["lr"]:
            # For learning rate, use logarithmic scale sampling (more practical)
            # Generate 5-8 values on log scale between min and max
            num_samples = min(8, max(5, int(np.log10(max_val / min_val) * 2)))
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            log_values = np.linspace(log_min, log_max, num_samples)
            result = [round(10**log_val, 6) for log_val in log_values]
            return result
        elif param_name in ["do"]:
            # For dropout, generate reasonable steps
            num_steps = 5
            step = (max_val - min_val) / num_steps
            result = []
            current = min_val
            while current <= max_val + step / 2:
                result.append(round(current, 2))
                current += step
            return result
        else:
            return list(range(min_val, max_val + 1))

    def _generate_range_with_step(self, min_val, max_val, step, param_name):
        """Generate parameter range with explicit step."""
        if (
            isinstance(step, float)
            or isinstance(min_val, float)
            or isinstance(max_val, float)
        ):
            result = []
            current = float(min_val)
            step = float(step)
            max_val = float(max_val)
            while current <= max_val + step / 2:
                result.append(round(current, 6))  # More precision for floats
                current += step
            return result
        else:
            return list(range(min_val, max_val + 1, step))

    def generate_param_combinations(self, param_specs):
        """Generate all parameter combinations for grid search."""
        param_names = list(param_specs.keys())
        param_values = list(param_specs.values())

        combinations = []
        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def run_optimization(self):
        """Run hyperparameter optimization using the most appropriate method."""
        self.util.debug(
            f"Starting optimization using {self.search_strategy} strategy with {self.metric} metric, nkululeko version {VERSION}"
        )

        param_specs = self.parse_optim_params()

        if not param_specs:
            self.util.error("No optimization parameters found in [OPTIM] section")
            return None, None, []

        # Always use manual optimization to ensure consistent evaluation pipeline
        # This prevents discrepancies between CV and final evaluation
        self.util.debug("Using manual optimization for consistent evaluation pipeline")
        return self._run_manual_optimization(param_specs)

    def _run_manual_optimization(self, param_specs):
        """Run manual grid search optimization with consistent evaluation pipeline."""
        combinations = self.generate_param_combinations(param_specs)

        if not combinations:
            self.util.error("No parameter combinations generated")
            return None, None, []

        self.util.debug(
            f"Starting manual optimization with {len(combinations)} parameter combinations"
        )

        # Check if we should use cross-validation or train-test split
        use_cv = self.search_strategy in ["grid_cv", "random_cv"] or (
            hasattr(self, 'use_cv_in_manual') and self.use_cv_in_manual
        )

        if use_cv:
            return self._run_manual_cv_optimization(combinations, param_specs)
        else:
            return self._run_manual_train_test_optimization(combinations)

    def _run_manual_train_test_optimization(self, combinations):
        """Run manual optimization using train-test split (matches final evaluation)."""
        best_result = None
        best_params = None
        best_score = -float("inf") if self.util.high_is_good() else float("inf")

        for i, params in enumerate(combinations):
            self.util.debug(f"Testing combination {i+1}/{len(combinations)}: {params}")

            self._update_config_with_params(params)

            try:
                result, last_epoch = self._run_single_experiment()
                score = result  # result.test is already a numeric value

                result_entry = {
                    "params": params.copy(),
                    "score": score,
                    "result": result,
                    "epoch": last_epoch,
                }
                self.results.append(result_entry)

                is_better = (self.util.high_is_good() and score > best_score) or (
                    not self.util.high_is_good() and score < best_score
                )

                if is_better:
                    best_score = score
                    best_result = result
                    best_params = params.copy()

                self.util.debug(f"Result: {result}, Score: {score}")

            except Exception as e:
                self.util.error(f"Failed with params {params}: {str(e)}")
                # Log the full traceback for debugging
                import traceback

                self.util.debug(f"Full traceback: {traceback.format_exc()}")
                continue

        self.util.debug("Optimization complete!")
        self.util.debug(f"Best parameters: {best_params}")
        self.util.debug(f"Best result: {best_result}")

        # Save results to file
        self.save_results()

        return best_params, best_result, self.results

    def _run_manual_cv_optimization(self, combinations, param_specs):
        """Run manual optimization using cross-validation."""
        import numpy as np
        from sklearn.model_selection import StratifiedKFold

        self.util.debug("Using cross-validation for optimization (may differ from final evaluation)")

        # Set up the experiment once to get the data
        import nkululeko.experiment as exp
        expr = exp.Experiment(self.config)
        expr.set_module("optim")
        expr.load_datasets()
        expr.fill_train_and_tests()
        expr.extract_feats()

        # Create stratified CV splits
        cv_splitter = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )

        best_result = None
        best_params = None
        best_score = -float("inf") if self.util.high_is_good() else float("inf")

        for i, params in enumerate(combinations):
            self.util.debug(f"Testing combination {i+1}/{len(combinations)}: {params}")

            # Run cross-validation for this parameter combination
            cv_scores = []
            
            try:
                for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(
                    expr.feats_train, expr.df_train[self.config["DATA"]["target"]]
                )):
                    self.util.debug(f"  Fold {fold+1}/{self.cv_folds}")
                    
                    # Create fold-specific data
                    fold_train_feats = expr.feats_train.iloc[train_idx]
                    fold_val_feats = expr.feats_train.iloc[val_idx]
                    fold_train_df = expr.df_train.iloc[train_idx]
                    fold_val_df = expr.df_train.iloc[val_idx]

                    # Update config with current parameters
                    self._update_config_with_params(params)

                    # Run experiment on this fold
                    fold_score = self._run_cv_fold(
                        fold_train_feats, fold_val_feats, 
                        fold_train_df, fold_val_df, params
                    )
                    cv_scores.append(fold_score)

                # Calculate mean CV score
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)

                result_entry = {
                    "params": params.copy(),
                    "score": mean_score,
                    "result": mean_score,
                    "cv_std": std_score,
                    "cv_scores": cv_scores,
                    "epoch": 0,
                }
                self.results.append(result_entry)

                is_better = (self.util.high_is_good() and mean_score > best_score) or (
                    not self.util.high_is_good() and mean_score < best_score
                )

                if is_better:
                    best_score = mean_score
                    best_result = mean_score
                    best_params = params.copy()

                self.util.debug(f"CV Score: {mean_score:.4f} ± {std_score:.4f}")

            except Exception as e:
                self.util.error(f"Failed with params {params}: {str(e)}")
                continue

        self.util.debug("Cross-validation optimization complete!")
        self.util.debug(f"Best parameters: {best_params}")
        self.util.debug(f"Best CV score: {best_result}")

        # Validate with final evaluation pipeline
        if best_params:
            validation_score = self._validate_best_params_standard_eval(best_params, expr)
            if validation_score is not None:
                self.util.debug(f"Cross-validation score: {best_result:.4f}")
                self.util.debug(f"Standard evaluation score: {validation_score:.4f}")
                score_diff = abs(best_result - validation_score)
                self.util.debug(f"Score difference: {score_diff:.4f}")
                
                if score_diff > 0.1:  # 10% difference threshold
                    self.util.debug("WARNING: Large discrepancy between CV and standard evaluation!")
                    self.util.debug("Consider using train-test optimization for more consistent results.")

        # Save results to file
        self.save_results()

        return best_params, best_result, self.results

    def _run_cv_fold(self, train_feats, val_feats, train_df, val_df, params):
        """Run a single cross-validation fold."""
        from nkululeko.modelrunner import Modelrunner

        # Create a temporary runner for this fold
        runner = Modelrunner(train_df, val_df, train_feats, val_feats, 0)
        runner._select_model(self.model_type)

        # Configure model with current parameters
        if self.model_type == "mlp":
            self._configure_mlp_model(runner.model, params)
        else:
            self._configure_traditional_model(runner.model, params)

        # Train and evaluate
        runner.model.train()
        reports = runner.model.predict()
        
        # Extract score based on metric
        return self._extract_score_from_report(reports)

    def _run_sklearn_optimization(self, param_specs):
        """Run optimization using scikit-learn's hyperparameter search methods with consistent data handling."""
        from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                             StratifiedKFold)

        # Import the actual experiment to get the model and data
        import nkululeko.experiment as exp

        # Set up the experiment
        expr = exp.Experiment(self.config)
        expr.set_module("optim")
        expr.load_datasets()
        expr.fill_train_and_tests()
        expr.extract_feats()

        # Apply the same balancing as the final evaluation
        original_train_feats = expr.feats_train.copy()
        original_train_df = expr.df_train.copy()
        
        if "FEATS" in self.config and "balancing" in self.config["FEATS"]:
            balancing_method = self.config["FEATS"]["balancing"]
            if balancing_method and balancing_method.lower() != "none":
                self.util.debug(f"Applying {balancing_method} balancing for optimization consistency")
                try:
                    from nkululeko.balance import DataBalancer
                    balancer = DataBalancer()
                    expr.feats_train, expr.df_train = balancer.balance_features(
                        expr.df_train, expr.feats_train, self.config["DATA"]["target"], balancing_method
                    )
                    self.util.debug(f"Balanced training data: {len(expr.feats_train)} samples")
                except Exception as e:
                    self.util.debug(f"Balancing failed: {e}, using original data")
                    expr.feats_train = original_train_feats
                    expr.df_train = original_train_df

        # Get the base model without hyperparameter tuning
        original_tuning_params = self.config.get(
            "MODEL", "tuning_params", fallback=None
        )
        if "MODEL" not in self.config:
            self.config.add_section("MODEL")

        # Temporarily disable tuning_params to get base model
        if original_tuning_params:
            self.config.remove_option("MODEL", "tuning_params")

        # Create a model instance using the modelrunner approach
        from nkululeko.modelrunner import Modelrunner

        runner = Modelrunner(
            expr.df_train, expr.df_test, expr.feats_train, expr.feats_test, 0
        )
        runner._select_model(self.model_type)
        base_clf = runner.model.clf

        # Restore original tuning_params if it existed
        if original_tuning_params:
            self.config.set("MODEL", "tuning_params", original_tuning_params)

        # Convert parameter specifications to sklearn format
        sklearn_params = self._convert_to_sklearn_params(param_specs)

        # Create stratified CV for consistent cross-validation
        cv = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )

        # Choose search strategy
        if self.search_strategy == "random":
            search = RandomizedSearchCV(
                base_clf,
                sklearn_params,
                n_iter=self.n_iter,
                cv=cv,  # Use stratified CV
                scoring=self._get_scoring_metric(),
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1,
            )
        elif self.search_strategy == "halving_random":
            try:
                from sklearn.model_selection import HalvingRandomSearchCV

                search = HalvingRandomSearchCV(
                    base_clf,
                    sklearn_params,
                    cv=cv,  # Use stratified CV
                    scoring=self._get_scoring_metric(),
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=1,
                )
            except ImportError:
                self.util.debug(
                    "HalvingRandomSearchCV not available, falling back to RandomizedSearchCV"
                )
                search = RandomizedSearchCV(
                    base_clf,
                    sklearn_params,
                    n_iter=self.n_iter,
                    cv=cv,  # Use stratified CV
                    scoring=self._get_scoring_metric(),
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=1,
                )
        elif self.search_strategy == "halving_grid":
            try:
                from sklearn.model_selection import HalvingGridSearchCV

                search = HalvingGridSearchCV(
                    base_clf,
                    sklearn_params,
                    cv=cv,  # Use stratified CV
                    scoring=self._get_scoring_metric(),
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=1,
                )
            except ImportError:
                self.util.debug(
                    "HalvingGridSearchCV not available, falling back to GridSearchCV"
                )
                search = GridSearchCV(
                    base_clf,
                    sklearn_params,
                    cv=cv,  # Use stratified CV
                    scoring=self._get_scoring_metric(),
                    n_jobs=-1,
                    verbose=1,
                )
        else:  # grid search (default)
            search = GridSearchCV(
                base_clf,
                sklearn_params,
                cv=cv,  # Use stratified CV
                scoring=self._get_scoring_metric(),
                n_jobs=-1,
                verbose=1,
            )

        self.util.debug(
            f"Starting {self.search_strategy} search with {len(sklearn_params)} parameters"
        )
        self.util.debug(f"Using stratified {self.cv_folds}-fold cross-validation")

        # Fit the search
        search.fit(expr.feats_train, expr.df_train[self.config["DATA"]["target"]])

        # Extract results
        best_params = search.best_params_
        best_score = search.best_score_

        # Convert results back to our format
        all_results = []
        for i, (params, score) in enumerate(
            zip(search.cv_results_["params"], search.cv_results_["mean_test_score"])
        ):
            result_entry = {
                "params": params,
                "score": score,
                "result": score,
                "epoch": 0,
            }
            all_results.append(result_entry)

        self.results = all_results

        self.util.debug("Optimization complete!")
        self.util.debug(f"Best parameters: {best_params}")
        self.util.debug(f"Best score: {best_score}")

        # Save results
        self.save_results()
        
        # Validate best parameters using standard nkululeko evaluation for consistency
        validation_score = self._validate_best_params_standard_eval(best_params, expr)
        if validation_score is not None:
            self.util.debug(f"Cross-validation score: {best_score:.4f}")
            self.util.debug(f"Standard evaluation score: {validation_score:.4f}")
            score_diff = abs(best_score - validation_score)
            self.util.debug(f"Score difference: {score_diff:.4f}")
            
            if score_diff > 0.1:  # 10% difference threshold
                self.util.debug("WARNING: Large discrepancy between CV and standard evaluation!")
                self.util.debug("This may indicate overfitting to CV folds or inconsistent data handling.")
                self.util.debug("Consider using manual optimization for more consistent results.")

        return best_params, best_score, all_results

    def _convert_to_sklearn_params(self, param_specs):
        """Convert our parameter specifications to sklearn format."""
        # Parameter name mapping from nkululeko names to sklearn names
        param_mapping = {
            # SVM parameters
            "C_val": "C",  # SVM regularization parameter
            "c_val": "C",  # Alternative lowercase version
            # KNN parameters
            "K_val": "n_neighbors",  # KNN number of neighbors
            "k_val": "n_neighbors",  # Alternative lowercase version
            "KNN_weights": "weights",  # KNN weights (uniform/distance)
            "knn_weights": "weights",  # Alternative lowercase version
        }
        
        sklearn_params = {}
        for param_name, values in param_specs.items():
            # Map parameter names to sklearn equivalents
            sklearn_param_name = param_mapping.get(param_name, param_name)
            
            if isinstance(values, list):
                sklearn_params[sklearn_param_name] = values
            else:
                # Convert single values to lists
                sklearn_params[sklearn_param_name] = [values]
        return sklearn_params

    def _get_scoring_metric(self):
        """Get the appropriate scoring metric for sklearn optimization."""
        # Create custom scorer for specificity if needed
        if self.metric == "specificity":
            from sklearn.metrics import make_scorer

            def specificity_score(y_true, y_pred):
                import numpy as np
                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(y_true, y_pred)
                if cm.shape[0] == 2:  # Binary classification
                    tn = cm[0, 0]
                    fp = cm[0, 1]
                    return tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:  # Multi-class: average specificity
                    specificities = []
                    for i in range(cm.shape[0]):
                        tn = np.sum(cm) - (
                            np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i]
                        )
                        fp = np.sum(cm[:, i]) - cm[i, i]
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                        specificities.append(specificity)
                    return np.mean(specificities)

            return make_scorer(specificity_score)

        # Standard scikit-learn metrics
        metric_map = {
            "uar": "balanced_accuracy",  # Unweighted Average Recall
            "accuracy": "accuracy",  # Standard accuracy
            "f1": "f1_macro",  # Macro-averaged F1
            "precision": "precision_macro",  # Macro-averaged precision
            "recall": "recall_macro",  # Macro-averaged recall
            "sensitivity": "recall_macro",  # Sensitivity = recall
        }

        if self.util.exp_is_classification():
            return metric_map.get(self.metric or "accuracy", "accuracy")
        else:
            # For regression tasks
            if self.metric in [
                "accuracy",
                "uar",
                "f1",
                "precision",
                "recall",
                "sensitivity",
                "specificity",
            ]:
                self.util.debug(
                    f"Warning: {self.metric} is not suitable for regression, using RMSE"
                )
            return "neg_root_mean_squared_error"

    def _update_config_with_params(self, params):
        """Update configuration with current parameter set."""
        self._ensure_model_section()

        if self.model_type == "mlp":
            self._update_mlp_params(params)
        else:
            self._update_traditional_ml_params(params)

    def _ensure_model_section(self):
        """Ensure MODEL section exists with basic configuration."""
        if "MODEL" not in self.config:
            self.config.add_section("MODEL")

        if "type" not in self.config["MODEL"]:
            self.config["MODEL"]["type"] = self.model_type

    def _update_mlp_params(self, params):
        """Update MLP-specific parameters."""
        if "nlayers" in params and "nnodes" in params:
            nlayers = params["nlayers"]
            nnodes = params["nnodes"]
            layers = {f"l{i+1}": nnodes for i in range(nlayers)}
            self.config["MODEL"]["layers"] = str(layers)

        if "lr" in params:
            self.config["MODEL"]["learning_rate"] = str(params["lr"])

        if "bs" in params:
            self.config["MODEL"]["batch_size"] = str(params["bs"])

        if "do" in params:
            self.config["MODEL"]["drop"] = str(params["do"])

        if "loss" in params:
            self.config["MODEL"]["loss"] = params["loss"]

    def _update_traditional_ml_params(self, params):
        """Update traditional ML parameters using tuning_params approach."""
        # For optimization, we set the specific parameter values directly
        # rather than using the tuning mechanism
        for param_name, param_value in params.items():
            self.config["MODEL"][param_name] = str(param_value)
        
        # Add random_state to model configuration for consistency
        if self.model_type in ["xgb", "xgr", "svm", "svr", "knn", "knn_reg", "tree", "tree_reg"]:
            self.config["MODEL"]["random_state"] = str(self.random_state)

    def _run_single_experiment(self):
        """Run a single experiment with current configuration."""
        import nkululeko.experiment as exp

        if "MODEL" not in self.config:
            self.config.add_section("MODEL")
        if "type" not in self.config["MODEL"]:
            self.config["MODEL"]["type"] = self.model_type

        expr = exp.Experiment(self.config)
        expr.set_module("optim")

        expr.load_datasets()

        expr.fill_train_and_tests()

        expr.extract_feats()

        expr.init_runmanager()

        reports, last_epochs = expr.run()
        result = expr.get_best_report(reports).result.test

        return result, int(min(last_epochs))

    def save_results(self, filepath=None):
        """Save optimization results to CSV file."""
        if not self.results:
            self.util.debug("No results to save")
            return

        if filepath is None:
            # Save in the results directory instead of current directory
            results_dir = self.util.get_path("res_dir")
            filepath = os.path.join(
                results_dir, f"optimization_results_{self.model_type}.csv"
            )

        import csv

        try:
            with open(filepath, "w", newline="") as csvfile:
                # Get all unique parameter names from all results
                param_names = set()
                for result in self.results:
                    param_names.update(result["params"].keys())
                param_names = sorted(list(param_names))

                fieldnames = param_names + ["score", "result", "epoch"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for result in self.results:
                    row = result["params"].copy()
                    row["score"] = result["score"]
                    row["result"] = result["result"]
                    row["epoch"] = result["epoch"]
                    writer.writerow(row)

            self.util.debug(f"Optimization results saved to {filepath}")
        except Exception as e:
            self.util.error(f"Failed to save results: {e}")

    def get_best_params(self):
        """Get the best parameters found during optimization."""
        if not self.results:
            return None

        best_result = None
        best_score = -float("inf") if self.util.high_is_good() else float("inf")

        for result in self.results:
            score = result["score"]
            is_better = (self.util.high_is_good() and score > best_score) or (
                not self.util.high_is_good() and score < best_score
            )
            if is_better:
                best_score = score
                best_result = result

        return best_result

    def get_recommended_ranges(self, param_name):
        """Get recommended parameter ranges for common hyperparameters."""
        recommendations = {
            "lr": [0.0001, 0.001, 0.01, 0.1],  # Log-scale discrete values
            "do": [0.1, 0.3, 0.5, 0.7],  # Common dropout rates
            "C_val": [0.1, 1.0, 10.0, 100.0],  # SVM regularization
            "c_val": [0.1, 1.0, 10.0, 100.0],  # SVM regularization (alternative)
            "K_val": [3, 5, 7, 9, 11],  # KNN neighbors
            "k_val": [3, 5, 7, 9, 11],  # KNN neighbors (alternative)
            "KNN_weights": ["uniform", "distance"],  # KNN weights
            "knn_weights": ["uniform", "distance"],  # KNN weights (alternative)
            "n_estimators": [50, 100, 200],  # XGB trees
            "max_depth": [3, 6, 9, 12],  # Tree depth
            "subsample": [0.6, 0.8, 1.0],  # XGB subsample
            "learning_rate": [0.01, 0.1, 0.3],  # XGB learning rate
        }
        return recommendations.get(param_name, None)

    def _validate_best_params_standard_eval(self, best_params, expr):
        """Validate the best parameters using standard nkululeko train-test evaluation."""
        try:
            # Set the model parameters to the best found values
            self._update_config_with_params(best_params)
            
            # Run a single experiment with these parameters using the standard approach
            result, _ = self._run_single_experiment()
            
            return result
        except Exception as e:
            self.util.debug(f"Standard validation failed: {e}")
            return None

    def _configure_mlp_model(self, model, params):
        """Configure MLP model with current parameters."""
        # Set MLP-specific parameters
        if hasattr(model, 'clf') and hasattr(model.clf, 'set_params'):
            model_params = {}
            
            # Map optimization parameters to model parameters
            if "lr" in params:
                model_params["learning_rate"] = params["lr"]
            if "do" in params:
                model_params["dropout"] = params["do"]
            if "bs" in params:
                model_params["batch_size"] = params["bs"]
                
            model.clf.set_params(**model_params)

    def _configure_traditional_model(self, model, params):
        """Configure traditional ML model with current parameters."""
        if hasattr(model, 'clf') and hasattr(model.clf, 'set_params'):
            # Map parameter names for different models
            param_mapping = {
                "C_val": "C",
                "c_val": "C", 
                "K_val": "n_neighbors",
                "k_val": "n_neighbors",
                "KNN_weights": "weights",
                "knn_weights": "weights",
            }
            
            model_params = {}
            for param_name, param_value in params.items():
                sklearn_param = param_mapping.get(param_name, param_name)
                model_params[sklearn_param] = param_value
                
            model.clf.set_params(**model_params)

    def _extract_score_from_report(self, reports):
        """Extract score from model prediction reports."""
        # This is a simplified version - you may need to adapt based on your report structure
        if isinstance(reports, dict):
            # Try to extract the metric we're optimizing for
            if self.metric in reports:
                return reports[self.metric]
            elif "test" in reports:
                return reports["test"]
            else:
                # Return the first numeric value found
                for key, value in reports.items():
                    if isinstance(value, (int, float)):
                        return value
        elif isinstance(reports, (int, float)):
            return reports
        else:
            # Fallback: assume it's a list and take the first element
            try:
                return reports[0] if hasattr(reports, '__getitem__') else 0.0
            except:
                return 0.0


def doit(config_file):
    """Run hyperparameter optimization experiment."""
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_file)

    optimizer = OptimizationRunner(config)

    # Start timing the optimization
    start_time = time.time()

    # Run optimization using the unified approach
    try:
        best_params, best_result, all_results = optimizer.run_optimization()
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None, None

    # Calculate optimization time
    end_time = time.time()
    optimization_time = end_time - start_time

    print("OPTIMIZATION COMPLETE")
    print(f"Best parameters: {best_params}")
    print(f"Best result: {best_result}")
    print(
        f"Optimization time: {optimization_time:.2f} seconds ({optimization_time/60:.2f} minutes)"
    )
    print("DONE")
    return best_params, best_result


def main():
    """Main entry point for optimization module."""
    parser = argparse.ArgumentParser(
        description="Run nkululeko hyperparameter optimization."
    )
    parser.add_argument("--version", action="version", version=f"Nkululeko {VERSION}")
    parser.add_argument(
        "--config", default="exp.ini", help="The optimization configuration file"
    )
    args = parser.parse_args()

    config_file = args.config
    doit(config_file)


if __name__ == "__main__":
    main()
