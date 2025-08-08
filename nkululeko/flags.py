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
    
    results = []
    
    for i, combo in enumerate(combinations, 1):
        print(f"\n=== Experiment {i}/{len(combinations)} ===")
        print(f"Parameters: {combo}")
        
        # Create a copy of the original config
        experiment_config = configparser.ConfigParser()
        experiment_config.read_dict(config._sections)
        
        # Ensure required sections exist
        if "MODEL" not in experiment_config:
            experiment_config.add_section("MODEL")
        if "FEATS" not in experiment_config:
            experiment_config.add_section("FEATS")
            
        # Copy no_reuse from DATA to FEATS section if it exists to ensure proper feature reuse
        if "DATA" in experiment_config and "no_reuse" in experiment_config["DATA"]:
            experiment_config["FEATS"]["no_reuse"] = experiment_config["DATA"]["no_reuse"]
        
        # Apply combination parameters
        for param, value in combo.items():
            if param == "models":
                experiment_config["MODEL"]["type"] = value
            elif param == "features":
                experiment_config["FEATS"]["type"] = f"['{value}']"
            elif param == "balancing":
                experiment_config["FEATS"]["balancing"] = value
            elif param == "scale":
                experiment_config["FEATS"]["scale"] = value
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
        
        # Write temporary config
        tmp_config = f"tmp_flags_{i}.ini"
        with open(tmp_config, "w") as tmp_file:
            experiment_config.write(tmp_file)
        
        try:
            # Run the experiment
            result, last_epoch = nkulu(tmp_config)
            results.append({
                "parameters": combo,
                "result": result,
                "last_epoch": last_epoch,
                "config_file": tmp_config
            })
            print(f"Result: {result}")
            
        except Exception as e:
            print(f"ERROR in experiment {i}: {e}")
            results.append({
                "parameters": combo,
                "result": None,
                "last_epoch": None,
                "error": str(e),
                "config_file": tmp_config
            })
        
        # Clean up temporary config
        if os.path.exists(tmp_config):
            os.remove(tmp_config)
    
    # Print summary
    print(f"\n=== SUMMARY OF {len(combinations)} EXPERIMENTS ===")
    valid_results = []
    for i, result in enumerate(results, 1):
        print(f"Experiment {i}: {result['parameters']}")
        if result.get('error'):
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Result: {result['result']}")
            if result['result'] is not None:
                valid_results.append(result)
    
    # Find and print best parameters
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['result'])
        print(f"\n=== BEST CONFIGURATION ===")
        print(f"Best Result: {best_result['result']}")
        print(f"Best Parameters:")
        for param, value in best_result['parameters'].items():
            print(f"  {param}: {value}")
        print(f"\nTo use these parameters, set in your config file:")
        print(f"[MODEL]")
        print(f"type = {best_result['parameters'].get('models', 'N/A')}")
        print(f"[FEATS]")
        print(f"type = ['{best_result['parameters'].get('features', 'N/A')}']")
        if 'balancing' in best_result['parameters']:
            print(f"balancing = {best_result['parameters']['balancing']}")
        if 'scale' in best_result['parameters']:
            print(f"scale = {best_result['parameters']['scale']}")
    else:
        print("\n=== NO VALID RESULTS FOUND ===")
    
    return results


def doit(cla):
    parser = argparse.ArgumentParser(description="Call the nkululeko framework with multiple parameter combinations.")
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
    parser.add_argument("--flags", action="store_true", help="Run multiple experiments based on FLAGS section")

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
