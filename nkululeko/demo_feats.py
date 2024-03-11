# demo_feats.py
# Test some features extracted

from nkululeko.experiment import Experiment
import configparser
from nkululeko.utils.util import Util
from nkululeko.constants import VERSION
import argparse
import os


def main(src_dir):
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
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
    module = "demo_feats"
    expr.set_module(module)
    util = Util(module)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version"
        f" {VERSION}"
    )

    if args.file is None and args.list is None:
        expr.demo_feats(None, False, args.outfile)
    else:
        if args.list is None:
            expr.demo_feats(args.file, False, args.outfile)
        else:
            expr.demo_feats(args.list, True, args.outfile)

    print("DONE")


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # use this if you want to state the config file path on command line
