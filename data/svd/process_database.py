# pre-processing script for SVD dataset


import argparse
from pathlib import Path
import pandas as pd


def process_database(args):
    # arguments
    # input_path = args.data_dir
    output_path = args.out_dir

    train_normal = pd.read_csv('train_normal.lst', header=None)
    train_pathology = pd.read_csv('train_pathology.lst', header=None)
    dev_normal = pd.read_csv('develop_normal.lst', header=None)
    dev_pathology = pd.read_csv('develop_pathology.lst', header=None)
    test_normal = pd.read_csv('test_normal.lst', header=None)
    test_pathology = pd.read_csv('test_pathology.lst', header=None)


    # add vowel (-a) and label (_n/_p) to filenames in the first column
    train_normal['file'] = train_normal[0].apply(lambda x: f"{x}-a_n.wav")
    train_normal['label'] = "n"
    train_pathology['file'] = train_pathology[0].apply(lambda x: f"{x}-a_p.wav")
    train_pathology['label'] = "p"
    train = pd.concat([train_normal, train_pathology])
    # remove other column except file and label
    train = train[['file', 'label']]
    train.to_csv(output_path/'svd_train.csv', index=False)

    # for dev and test sets
    dev_normal['file'] = dev_normal[0].apply(lambda x: f"{x}-a_n.wav")
    dev_normal['label'] = "n"
    dev_pathology['file'] = dev_pathology[0].apply(lambda x: f"{x}-a_p.wav")
    dev_pathology['label'] = "p"
    dev = pd.concat([dev_normal, dev_pathology])
    dev = dev[['file', 'label']]
    dev.to_csv(output_path/'svd_dev.csv', index=False)
    
    test_normal['file'] = test_normal[0].apply(lambda x: f"{x}-a_n.wav")
    test_normal['label'] = "n"
    test_pathology['file'] = test_pathology[0].apply(lambda x: f"{x}-a_p.wav")
    test_pathology['label'] = "p"
    test = pd.concat([test_normal, test_pathology])
    dev = test[['file', 'label']]
    test.to_csv(output_path/'svd_test.csv', index=False)

    # print number of train, dev and test samples
    print(f"Number of train samples: {train.shape[0]}")
    print(f"Number of dev samples: {dev.shape[0]}")
    print(f"Number of test samples: {test.shape[0]}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=".",
        help="Path to store processed data")
    args = parser.parse_args()
    process_database(args)
