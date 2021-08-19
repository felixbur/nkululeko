# main.py
# Demonstration code to use the ML-experiment framework

import sys

import experiment as exp
import dataset as ds
import configparser
from emodb import Emodb

def main(config_file):
    # for two sexes
    sexes = ['female', 'male']

    for s in sexes:
        # load one configuration per experiment
        config = configparser.ConfigParser()
        config.read(config_file)
        # set the sex
        config['DATA']['sex'] = s 
        config['EXP']['name'] = config['EXP']['name']+'_'+s

        # create a new experiment
        expr = exp.Experiment(config)
        print(f'running {expr.name}')

        # load the data
        expr.load_datasets()

        # split into train and test
        expr.fill_train_and_tests()
        print(f'train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}')

        # extract spectra
        expr.extract_feats()

        # initialize a run manager
        expr.init_runmanager()

        # run the experiment
        expr.run()


if __name__ == "__main__":
    main('/home/fburkhardt/ResearchProjects/nkululeko/exp_bdtg_cnn.ini')# sys.argv[1])
