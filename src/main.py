import sys

import experiment as exp
import dataset as ds
import configparser
from emodb import Emodb

def main():
    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read('../exp_test.ini')

    # create a new experiment
    expr = exp.Experiment('my_cool_experiment', config)
    print(expr.name)

    # create a dataset
    data = Emodb(config)
    expr.add_dataset(data)
    for d in expr.datasets:
        print(d.name)
        print(d.config['DATA'][d.name])
        d.load()
        print(d.df.shape)

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
    main()
