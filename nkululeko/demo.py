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

import pandas as pd
from transformers import pipeline

import nkululeko.glob_conf as glob_conf
from nkululeko.constants import VERSION
from nkululeko.experiment import Experiment
from nkululeko.utils.util import Util


def main():
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

    def print_pipe(files, outfile):
        """Prints the pipeline output for a list of files, and optionally writes the results to an output file.

        Args:
            files (list): A list of file paths to process through the pipeline.
            outfile (str, optional): The path to an output file to write the pipeline results to.

        Returns:
            None
        """
        results = []
        for file in files:
            result = pipe(file, top_k=1)
            if result[0]["score"] != result[0]["score"]:  # Check for NaN
                print(f"ERROR: NaN value in pipeline output for file: {file}")
            else:
                results.append(f"{file}, {result[0]['label']}")
        print("\n".join(results))

        if outfile is not None:
            with open(outfile, "w") as f:
                f.write("\n".join(results))

    if util.get_model_type() == "finetune":
        model_path = os.path.join(util.get_exp_dir(), "models", "run_0", "torch")
        pipe = pipeline("audio-classification", model=model_path)
        if args.file is not None:
            print_pipe([args.file], args.outfile)
        elif args.list is not None:
            # read audio files from list
            print(f"Reading files from {args.list}")
            list_file = pd.read_csv(args.list, header="infer")
            files = list_file.iloc[:, 0].tolist()
            print_pipe(files, args.outfile)
        elif args.folder is not None:
            # read audio files from folder
            from nkululeko.utils.files import find_files

            files = find_files(args.folder, relative=True, ext=["wav", "mp3"])
            print_pipe(files, args.outfile)
        else:
            print("ERROR: input mic currently is not supported for finetuning")
        return

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
    main()
