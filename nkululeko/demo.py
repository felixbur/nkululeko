# demo.py
# Demonstration code to use the ML-experiment framework
# Test the loading of a previously trained model and demo mode
# needs the project config file to run before
"""This script is used to test the loading of a previously trained model.

And run it in demo mode.
It requires the project config file to be run before.

Usage:  
python -m nkululeko.demo [--config CONFIG] [--file FILE] [--list LIST] [--folder FOLDER] [--outfile OUTFILE]  

Options:   \n
--config CONFIG     The base configuration file (default: exp.ini) \n  
--file FILE         A file that should be processed (16kHz mono wav) \n  
--list LIST         A file with a list of files, one per line, that should be processed (16kHz mono wav) \n  
--folder FOLDER     A name of a folder where the files within the list are in   (default: ./) \n   
--outfile OUTFILE   A filename to store the results in CSV  (default: None)  
"""
import argparse
import configparser
import os

from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
import nkululeko.glob_conf as glob_conf
from nkululeko.utils.util import Util


def main(src_dir):
    parser = argparse.ArgumentParser(description="Call the nkululeko DEMO framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    parser.add_argument(
        "--file", help="A file that should be processed (16kHz mono wav)"
    )
    parser.add_argument(
        "--list",
        help=(
            "A file with a list of files, one per line, that should be"
            " processed (16kHz mono wav)"
        ),
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--folder",
        help=("A name of a folder where the files within the list are in."),
        nargs="?",
        default="./",
    )
    parser.add_argument(
        "--outfile",
        help=("A filename to store the results in CSV"),
        nargs="?",
        default=None,
    )
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config
    else:
        config_file = f"{src_dir}/exp.ini"

    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = Experiment(config)
    module = "demo"
    expr.set_module(module)
    util = Util(module)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version"
        f" {VERSION}"
    )

    # load the experiment
    expr.load(f"{util.get_save_name()}")
    if args.folder is not None:
        glob_conf.config["DATA"]["test_folder"] = args.folder
    if args.file is None and args.list is None:
        expr.demo(None, False, args.outfile)
    else:
        if args.list is None:
            expr.demo(args.file, False, args.outfile)
        else:
            expr.demo(args.list, True, args.outfile)

    print("DONE")


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # use this if you want to state the config file path on command line
