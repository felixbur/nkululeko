#!/usr/bin/env python3

import argparse
import configparser
import sys
import time
from pathlib import Path

from nkululeko.constants import VERSION
import nkululeko.glob_conf as glob_conf
from nkululeko.optimizationrunner import OptimizationRunner


def doit(config_file):
    """Run hyperparameter optimization experiment."""
    config_path = Path(config_file)
    if not config_path.is_file():
        print(f"ERROR: no such file: {config_file}")
        sys.exit(1)

    config = configparser.ConfigParser()
    config.read(config_path)
    glob_conf.init_config(config)

    optimizer = OptimizationRunner(config)

    start_time = time.time()

    try:
        best_params, best_result, _ = optimizer.run_optimization()
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None, None

    optimization_time = time.time() - start_time

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

    doit(args.config)


if __name__ == "__main__":
    main()
