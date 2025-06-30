#!/usr/bin/env python3

import argparse
import ast
import configparser
import itertools
import os

from nkululeko.constants import VERSION
from nkululeko.utils.util import Util


class OptimizationRunner:
    """Hyperparameter optimization runner for nkululeko experiments."""

    def __init__(self, config):
        self.config = config
        self.util = Util("optim")
        self.results = []

    def parse_optim_params(self):
        """Parse OPTIM section parameters into search spaces."""
        if "OPTIM" not in self.config:
            self.util.error("No [OPTIM] section found in configuration")

        optim_config = self.config["OPTIM"]
        self.model_type = optim_config.get("model", "mlp")

        param_specs = {}
        for key, value in optim_config.items():
            if key == "model":
                continue
            param_specs[key] = self._parse_param_spec(key, value)

        return param_specs

    def _parse_param_spec(self, param_name, param_value):
        """Parse individual parameter specification."""
        try:
            parsed = ast.literal_eval(param_value)
        except (ValueError, SyntaxError):
            if isinstance(param_value, str):
                return [param_value]
            return param_value

        if isinstance(parsed, tuple):
            if len(parsed) == 2:
                return self._generate_range(parsed[0], parsed[1], param_name)
            elif len(parsed) == 3:
                return self._generate_range_with_step(
                    parsed[0], parsed[1], parsed[2], param_name
                )
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
        elif param_name in ["lr", "do"]:
            result = []
            current = min_val
            while current <= max_val + 0.0001:
                result.append(round(current, 4))
                current += 0.0001
            return result
        else:
            return list(range(min_val, max_val + 1))

    def _generate_range_with_step(self, min_val, max_val, step, param_name):
        """Generate parameter range with explicit step."""
        if isinstance(step, float) or isinstance(min_val, float):
            result = []
            current = min_val
            while current <= max_val + step/2:
                result.append(round(current, 4))
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
        """Run hyperparameter optimization."""
        self._ensure_model_section()
        
        param_specs = self.parse_optim_params()
        combinations = self.generate_param_combinations(param_specs)

        self.util.debug(
            f"Starting optimization with {len(combinations)} parameter combinations"
        )

        best_result = None
        best_params = None
        best_score = -float('inf') if self.util.high_is_good() else float('inf')

        for i, params in enumerate(combinations):
            self.util.debug(f"Testing combination {i+1}/{len(combinations)}: {params}")

            self._update_config_with_params(params)

            try:
                result, last_epoch = self._run_single_experiment()
                score = float(result.split()[1]) if isinstance(result, str) else result

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
                self.util.debug(f"Failed with params {params}: {e}")
                continue

        self.util.debug(f"Optimization complete!")
        self.util.debug(f"Best parameters: {best_params}")
        self.util.debug(f"Best result: {best_result}")

        return best_params, best_result, self.results

    def _update_config_with_params(self, params):
        """Update configuration with current parameter set."""
        if "MODEL" not in self.config:
            self.config.add_section("MODEL")

        self.config["MODEL"]["type"] = self.model_type

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
        tuning_param_names = []
        for param_name, param_value in params.items():
            self.config["MODEL"][param_name] = str([param_value])
            tuning_param_names.append(param_name)

        self.config["MODEL"]["tuning_params"] = str(tuning_param_names)
        self.config["MODEL"]["scoring"] = "recall_macro"

    def _run_single_experiment(self):
        """Run a single experiment with current configuration."""
        import nkululeko.experiment as exp
        
        expr = exp.Experiment(self.config)
        expr.set_module("optim")
        
        expr.load_datasets()
        
        expr.fill_train_and_tests()
        
        expr.extract_feats()
        
        expr.init_runmanager()
        
        reports, last_epochs = expr.run()
        result = expr.get_best_report(reports).result.test
        
        return result, int(min(last_epochs))


def doit(config_file):
    """Run hyperparameter optimization experiment."""
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    config = configparser.ConfigParser()
    config.read(config_file)
    
    optimizer = OptimizationRunner(config)
    
    best_params, best_result, all_results = optimizer.run_optimization()
    
    print("OPTIMIZATION COMPLETE")
    print(f"Best parameters: {best_params}")
    print(f"Best result: {best_result}")
    
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
