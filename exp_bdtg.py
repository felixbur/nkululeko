# main.py
# Demonstration code to use the ML-experiment framework

import sys
sys.path.append("/home/felix/data/research/nkululeko/src")

import experiment as exp
import dataset as ds
import configparser
import glob_conf
from util import Util
import matplotlib.pyplot as plt
import numpy as np

def main(config_file):
    util = Util()

    # red the config file
    config = configparser.ConfigParser()
    config.read(config_file)

    # create a new experiment
    expr = exp.Experiment(config)
    util.debug(f'running {expr.name}')

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f'train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}')

    # extract features
    expr.extract_feats()
    util.debug(f'train feats shape : {expr.feats_train.df.shape}, test feats shape:{expr.feats_test.df.shape}')

    # initialize a run manager
    expr.init_runmanager()

    # run the experiment
    mses_dev, mses_train, losses = expr.run()

    # plot the results
    fig_dir = util.get_path('fig_dir')
    plot_name = glob_conf.config['FEATS']['type']+'_mlp_results.png'
    plot_results(mses_dev, mses_train, losses, fig_dir+plot_name)

    print('DONE')

def plot_results(mses_dev, mses_train, losses, name):
    # do a plot per run
    # scale the losses so they fit on the picture
    losses = np.asarray(losses)/2
    plt.figure(dpi=200)
    plt.plot(mses_train, 'green', label='train set') 
    plt.plot(mses_dev, 'red', label='dev set')
    plt.plot(losses, 'grey', label='losses/2')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(name)
    plt.close()

if __name__ == "__main__":
    main('/home/felix/data/research/nkululeko/exp_bdtg_mlp.ini')# sys.argv[1])
