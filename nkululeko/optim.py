#!/usr/bin/env python3

import argparse
import configparser
import os
import sys
import time

from nkululeko.constants import VERSION

# Import the OptimizationRunner class from the dedicated module
from nkululeko.optimizationrunner import OptimizationRunner


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

    optimizer.util.debug(
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
