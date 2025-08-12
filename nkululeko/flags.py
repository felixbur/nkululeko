import argparse
import ast
import configparser
import itertools
import os
import os.path
import sys

from nkululeko.nkululeko import doit as nkulu
from nkululeko.testing import do_it as test_mod


def run_flags_experiments(config_file):
    """Run multiple experiments based on FLAGS section combinations."""
    import time

    # Start timing the flags experiments
    start_time = time.time()

    config = configparser.ConfigParser()
    config.read(config_file)

    # Check if FLAGS section exists
    if "FLAGS" not in config:
        print("ERROR: No [FLAGS] section found in configuration")
        return []

    flags_config = config["FLAGS"]

    # Parse flag parameters
    flag_params = {}
    for key, value in flags_config.items():
        try:
            flag_params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            flag_params[key] = [value]

    print(f"Flag parameters found: {flag_params}")

    # Generate all combinations
    param_names = list(flag_params.keys())
    param_values = list(flag_params.values())

    combinations = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)

    print(f"Running {len(combinations)} experiment combinations...")

    # Optimize feature extraction: load data and extract features ONCE
    print("Setting up experiment and extracting features (once for all experiments)...")
    import nkululeko.experiment as exp

    # Set up the config with the first feature type for feature extraction
    setup_config = configparser.ConfigParser()
    for section_name in config.sections():
        setup_config.add_section(section_name)
        for key, value in config.items(section_name):
            setup_config.set(section_name, key, value)

    # Set a valid feature type for initial extraction
    if "features" in flag_params and flag_params["features"]:
        first_feature = flag_params["features"][0]
        if "FEATS" not in setup_config:
            setup_config.add_section("FEATS")
        setup_config["FEATS"]["type"] = f"['{first_feature}']"

    # Create base experiment to load data and extract features
    base_experiment = exp.Experiment(setup_config)
    base_experiment.set_module("flags")

    try:
        base_experiment.load_datasets()
        base_experiment.fill_train_and_tests()
        base_experiment.extract_feats()
        print(f"Features extracted once: {base_experiment.feats_train.shape}")
    except Exception as e:
        print(f"ERROR during feature extraction setup: {e}")
        return []

    results = []

    for i, combo in enumerate(combinations, 1):
        print(f"\n=== Experiment {i}/{len(combinations)} ===")
        print(f"Parameters: {combo}")

        try:
            # Run experiment with current parameters (reusing extracted features)
            result, last_epoch = _run_single_flags_experiment(
                base_experiment, combo, config
            )
            results.append(
                {
                    "parameters": combo,
                    "result": result,
                    "last_epoch": last_epoch,
                    "config_file": None,  # No temp file needed
                }
            )
            print(f"Result: {result}")

        except Exception as e:
            print(f"ERROR in experiment {i}: {e}")
            results.append(
                {
                    "parameters": combo,
                    "result": None,
                    "last_epoch": None,
                    "error": str(e),
                    "config_file": None,
                }
            )

    # Print summary
    print(f"\n=== SUMMARY OF {len(combinations)} EXPERIMENTS ===")
    valid_results = []
    for i, result in enumerate(results, 1):
        print(f"Experiment {i}: {result['parameters']}")
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Result: {result['result']}")
            if result["result"] is not None:
                valid_results.append(result)

    # Find and print best parameters
    if valid_results:
        best_result = max(valid_results, key=lambda x: x["result"])
        print("\n=== BEST CONFIGURATION ===")
        print(f"Best Result: {best_result['result']}")
        print("Best Parameters:")
        for param, value in best_result["parameters"].items():
            print(f"  {param}: {value}")
        print("\nTo use these parameters, set in your config file:")
        print("[MODEL]")
        print(f"type = {best_result['parameters'].get('models', 'N/A')}")
        print("[FEATS]")
        print(f"type = ['{best_result['parameters'].get('features', 'N/A')}']")
        if "balancing" in best_result["parameters"]:
            print(f"balancing = {best_result['parameters']['balancing']}")
        if "scale" in best_result["parameters"]:
            print(f"scale = {best_result['parameters']['scale']}")
    else:
        print("\n=== NO VALID RESULTS FOUND ===")

    # Calculate and print total timing
    end_time = time.time()
    total_time = end_time - start_time
    print(
        f"\nFlags experiments time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    print("DONE")

    return results


def _run_single_flags_experiment(base_experiment, combo, config):
    """Run a single experiment with given parameters, reusing already extracted features."""
    import nkululeko.experiment as exp

    # Create a copy of the config for this experiment
    experiment_config = configparser.ConfigParser()
    for section_name in config.sections():
        experiment_config.add_section(section_name)
        for key, value in config.items(section_name):
            experiment_config.set(section_name, key, value)

    # Ensure required sections exist (they might be commented out in the original config)
    if "MODEL" not in experiment_config:
        experiment_config.add_section("MODEL")
    if "FEATS" not in experiment_config:
        experiment_config.add_section("FEATS")

    # Apply combination parameters to the config
    for param, value in combo.items():
        if param == "models":
            experiment_config["MODEL"]["type"] = value  # Set model type directly
        elif param == "features":
            experiment_config["FEATS"]["type"] = (
                f"['{value}']"  # Set feature type correctly
            )
        elif param == "balancing":
            # Only set balancing if not "none"
            if value.lower() != "none":
                experiment_config["FEATS"]["balancing"] = value
            # If "none", don't set balancing parameter (use default behavior)
        elif param == "scale":
            # Only set scale if not "none"
            if value.lower() != "none":
                experiment_config["FEATS"]["scale"] = value
            # If "none", don't set scale parameter (use default behavior)
        else:
            # Handle other custom parameters
            if param.startswith("model_"):
                experiment_config["MODEL"][param[6:]] = str(value)
            elif param.startswith("feats_"):
                experiment_config["FEATS"][param[6:]] = str(value)
            elif param.startswith("exp_"):
                experiment_config["EXP"][param[4:]] = str(value)

    # Create experiment name with parameters
    base_name = experiment_config["EXP"]["name"]
    param_suffix = "_".join([f"{k}_{v}" for k, v in combo.items()])
    experiment_config["EXP"]["name"] = f"{base_name}_{param_suffix}"

    # Update the experiment config - create a new experiment object with this config
    # to avoid modifying the shared base_experiment
    experiment = exp.Experiment(experiment_config)
    experiment.set_module("flags")

    # Copy the already extracted data and features from base_experiment
    experiment.df_train = base_experiment.df_train.copy()
    experiment.df_test = base_experiment.df_test.copy()
    experiment.feats_train = base_experiment.feats_train.copy()
    experiment.feats_test = base_experiment.feats_test.copy()

    # Initialize run manager with the updated config
    experiment.init_runmanager()

    # Run the experiment (this will use the already extracted features)
    reports, last_epochs = experiment.run()
    result = experiment.get_best_report(reports).result.test

    return result, int(min(last_epochs))


def doit(cla):
    parser = argparse.ArgumentParser(
        description="Call the nkululeko framework with multiple parameter combinations."
    )
    parser.add_argument("--config", help="The base configuration")
    parser.add_argument("--mod", default="nkulu", help="Which nkululeko module to call")
    parser.add_argument("--data", help="The databases", nargs="*", action="append")
    parser.add_argument(
        "--label", nargs="*", help="The labels for the target", action="append"
    )
    parser.add_argument(
        "--tuning_params", nargs="*", help="parameters to be tuned", action="append"
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        help="layer config for mlp, e.g. l1:128 ",
        action="append",
    )
    parser.add_argument("--model", default="xgb", help="The model type")
    parser.add_argument("--feat", default="['os']", help="The feature type")
    parser.add_argument("--set", help="The opensmile set")
    parser.add_argument("--target", help="The target designation")
    parser.add_argument("--epochs", help="The number of epochs")
    parser.add_argument("--runs", help="The number of runs")
    parser.add_argument("--learning_rate", help="The learning rate")
    parser.add_argument("--drop", help="The dropout rate [0:1]")
    parser.add_argument("--balancing", help="The balancing method")
    parser.add_argument("--scale", help="The scaling method")
    parser.add_argument(
        "--flags",
        action="store_true",
        help="Run multiple experiments based on FLAGS section",
    )

    args = parser.parse_args(cla)

    if args.config is not None:
        config_file = args.config
    else:
        print("ERROR: need config file")
        quit(-1)

    if args.mod is not None:
        nkulu_mod = args.mod

    # test if config is there
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file {config_file}")
        return None, None

    # If --flags is specified or FLAGS section exists, run multiple experiments
    config = configparser.ConfigParser()
    config.read(config_file)

    if args.flags or "FLAGS" in config:
        return run_flags_experiments(config_file)

    # Original single experiment logic
    # Ensure required sections exist
    if "MODEL" not in config:
        config.add_section("MODEL")

    # fill the config with command line arguments
    if args.data is not None:
        databases = []
        for t in args.data:
            databases.append(t[0])
        print(f"got databases: {databases}")
        config["DATA"]["databases"] = str(databases)
    if args.label is not None:
        labels = []
        for label in args.label:
            labels.append(label[0])
        print(f"got labels: {labels}")
        config["DATA"]["labels"] = str(labels)
    if args.tuning_params is not None:
        tuning_params = []
        for tp in args.tuning_params:
            tuning_params.append(tp[0])
        config["MODEL"]["tuning_params"] = str(tuning_params)
    if args.layers is not None:
        config["MODEL"]["layers"] = args.layers[0][0]
    if args.target is not None:
        config["DATA"]["target"] = args.target
    if args.epochs is not None:
        config["EXP"]["epochs"] = args.epochs
    if args.runs is not None:
        config["EXP"]["runs"] = args.runs
    if args.learning_rate is not None:
        config["MODEL"]["learning_rate"] = args.learning_rate
    if args.drop is not None:
        config["MODEL"]["drop"] = args.drop
    if args.model is not None:
        config["MODEL"]["type"] = args.model
    if args.feat is not None:
        config["FEATS"]["type"] = f"['{args.feat}']"
    if args.set is not None:
        config["FEATS"]["set"] = args.set
    if args.balancing is not None:
        config["FEATS"]["balancing"] = args.balancing
    if args.scale is not None:
        config["FEATS"]["scale"] = args.scale

    tmp_config = "tmp.ini"
    with open(tmp_config, "w") as tmp_file:
        config.write(tmp_file)

    result, last_epoch = 0, 0
    if nkulu_mod == "nkulu":
        result, last_epoch = nkulu(tmp_config)
    elif nkulu_mod == "test":
        result, last_epoch = test_mod(tmp_config, "test_results.csv")
    else:
        print(f"ERROR: unknown module: {nkulu_mod}, should be [nkulu | test]")

    # Clean up
    if os.path.exists(tmp_config):
        os.remove(tmp_config)

    return result, last_epoch


if __name__ == "__main__":
    cla = sys.argv
    cla.pop(0)
    doit(cla)
