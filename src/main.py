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
    expr = exp.Experiment('my cool experiment')
    print(expr.name)

    # create a dataset
    data = Emodb(config)
    expr.add_dataset(data)
    for d in expr.datasets:
        print(d.name)
        print(d.config['data'][d.name])
        d.load()
        print(d.df.shape)
    expr.fill_train_and_tests()
    print(f'train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}')


if __name__ == "__main__":
    main()
