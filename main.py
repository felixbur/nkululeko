# main.py
# Demonstration code to use the ML-experiment framework


from os import lseek
import sys
sys.path.append("/home/fburkhardt/ResearchProjects/nkululeko/src")
import experiment as exp
import dataset as ds
import configparser
from emodb import Emodb

def main(config_file):
    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = exp.Experiment(config)
    print(f'running {expr.name}')

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    print(f'train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}')

    # extract features
    expr.extract_feats()
    print(f'train feats shape : {expr.feats_train.df.shape}, test feats shape:{expr.feats_test.df.shape}')

    # initialize a run manager
    expr.init_runmanager()

    # run the experiment
    expr.run()


if __name__ == "__main__":
    main('/home/fburkhardt/ResearchProjects/nkululeko/exp_xgb.ini') #sys.argv[1])
