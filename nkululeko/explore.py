# explore.py
# explore the feature sets

from nkululeko.experiment import Experiment
import configparser
from nkululeko.utils.util import Util
from nkululeko.constants import VERSION
import argparse
import os


def main(src_dir):
    parser = argparse.ArgumentParser(description="Call the nkululeko framework.")
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
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
    module = "explore"
    expr.set_module(module)
    util = Util(module)
    util.debug(
        f"running {expr.name} from config {config_file}, nkululeko version"
        f" {VERSION}"
    )

    if util.config_val("EXP", "no_warnings", False):
        import warnings

        warnings.filterwarnings("ignore")

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f"train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}")

    plot_feats = eval(util.config_val("EXPL", "feature_distributions", "False"))
    tsne = eval(util.config_val("EXPL", "tsne", "False"))
    scatter = eval(util.config_val("EXPL", "scatter", "False"))
    spotlight = eval(util.config_val("EXPL", "spotlight", "False"))
    model_type = util.config_val("EXPL", "model", False)
    plot_tree = eval(util.config_val("EXPL", "plot_tree", "False"))
    needs_feats = False
    if plot_feats or tsne or scatter or model_type or plot_tree:
        # these investigations need features to explore
        expr.extract_feats()
        needs_feats = True
    # explore
    expr.analyse_features(needs_feats)
    expr.store_report()
    print("DONE")


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd)  # use this if you want to state the config file path on command line
