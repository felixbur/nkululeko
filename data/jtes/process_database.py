#!/usr/bin/env python3

import argparse
import glob
import os

import pandas as pd


def main(data_dir):
    files = glob.glob(os.path.join(data_dir + 'wav/*/*/', '*.wav'))
    files.sort()

    data_train = []
    data_dev = []
    data_test = []
    
    for file in files:
        # processing file
        # print("Processing... ", file)
        lab_str = os.path.basename(os.path.dirname(file))
        # use text 1-30 as training, the rest as test        
        if int(os.path.basename(file)[8:10]) in range(1, 31):
            data_train.append({
                "file": file,
                "emotion": lab_str,
                "speaker": int(os.path.basename(file)[1:3]),
            })
    
        # use text 31-40 as dev, the rest as test
        elif int(os.path.basename(file)[8:10]) in range(31, 41):
            data_dev.append({
                "file": file,
                "emotion": lab_str,
                "speaker": int(os.path.basename(file)[1:3]),
            })
        
        else:
            data_test.append({
                "file": file,
                "emotion": lab_str,
                "speaker": int(os.path.basename(file)[1:3]),
            })
    # check length of data
    print(f"Lenghth train, dev, test: {len(data_train)}, {len(data_dev)}, {len(data_test)}")

    # save as csv
    pd.DataFrame(data_train).to_csv(f"jtes_ti_train.csv", index=False)
    pd.DataFrame(data_dev).to_csv(f"jtes_ti_dev.csv", index=False)
    pd.DataFrame(data_test).to_csv(f"jtes_ti_test.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./jtes_v1.1/')
    args = parser.parse_args()
    main(args.data_dir)
