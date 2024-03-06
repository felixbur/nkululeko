# aug_train.py
# train with augmentations
import ast
import os.path
import numpy as np
import configparser
import argparse
import nkululeko.experiment as exp
from nkululeko.utils.util import Util
from nkululeko.constants import VERSION
import nkululeko.glob_conf as glob_conf
from nkululeko.augment import doit as augment


def doit(config_file):
    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f"ERROR: no such file: {config_file}")
        exit()

    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = exp.Experiment(config)
    module = "aug_train"
    expr.set_module(module)
    util = Util(module)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version"
        f" {VERSION}"
    )

    if util.config_val("EXP", "no_warnings", False):
        import warnings

        warnings.filterwarnings("ignore")

    augmentings = util.config_val("AUGMENT", "augment", False)
    if not augmentings:
        util.error("no augmentation method specified")
    augmentings = ast.literal_eval(augmentings)
    augmentings = "_".join(augmentings)
    result_file = f"augmented_{augmentings}.csv"

    glob_conf.config["DATA"]["no_reuse"] = "True"
    glob_conf.config["FEATS"]["no_reuse"] = "True"
    glob_conf.config["AUGMENT"]["sample_selection"] = "train"
    glob_conf.config["AUGMENT"]["result"] = f"./{result_file}"
    tmp_config = "tmp.ini"
    with open(tmp_config, "w") as config_file:
        glob_conf.config.write(config_file)
    augment(tmp_config)
    databases = ast.literal_eval(config["DATA"]["databases"])
    aug_name = f"aug_{augmentings}"
    databases.append(aug_name)
    glob_conf.config["DATA"]["databases"] = str(databases)
    glob_conf.config["DATA"][aug_name] = f"{util.get_exp_dir()}/{result_file}"
    glob_conf.config["DATA"][f"{aug_name}.type"] = "csv"
    glob_conf.config["DATA"][f"{aug_name}.rename_speakers"] = "True"
    glob_conf.config["DATA"][f"{aug_name}.split_strategy"] = "train"
    util.set_config(glob_conf.config)
    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}")

    # extract features
    expr.extract_feats()

    # initialize a run manager
    expr.init_runmanager()

    # run the experiment
    reports, last_epochs = expr.run()
    result = expr.get_best_report(reports).result.test
    expr.store_report()
    print("DONE")
    return result, int(np.asarray(last_epochs).min())


def main(src_dir):
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config
    else:
        config_file = f"{src_dir}/exp.ini"
    doit(config_file)


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # use this if you want to state the config file path on command line
