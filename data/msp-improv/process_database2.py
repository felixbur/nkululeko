# script to MSP-IMPROV label (ID, EmotionCategory, A, V, D, N)
# and split into train, dev, test, provided for Nkululeko
# bagustris@yahoo.com, 2023-08-22


import os
import pandas as pd
import csv
import argparse

train_list = []
dev_list = []
test_list = []

def main(data_path):
    with open(data_path + 'Evaluation.txt') as f:
        for line in f:
            if line[:3] == 'UTD':
                label = line.split(';')
               # Allocate Session6 for test
                filename = 'MSP-'+ label[0][4:-4]
                wavfile = os.path.join(
                    data_path, 
                    'session' + str(label[0].split('-')[3][-1]),
                    label[0].split('-')[2],
                    label[0].split('-')[4],
                    filename + '.wav')
                # gender information
                g = label[0].split('-')[3][0]
                # speaker information
                s = label[0].split('-')[3]
                emo_cat = label[1].strip()
                a = float(label[2][3:])
                v = float(label[3][3:])
                d = float(label[4][3:])
                n = float(label[5][3:])
                data = [wavfile, g, s, emo_cat, v, a, d, n]
                if label[0].split('-')[3][-2:] == '06':
                    test_list.append(data)
                elif label[0].split('-')[3][-2:] == '05':
                    dev_list.append(data)
                else:
                    train_list.append(data)

    test_list_df = pd.DataFrame(test_list).fillna(3)
    test_list_df.to_csv('improv_train.csv', index=False, header=['file', 'gender', 'speaker', 'emotion', 'valence', 'arousal', 'dominance', 'naturalness'])

    dev_list_df = pd.DataFrame(dev_list).fillna(3)
    dev_list_df.to_csv('improv_dev.csv', index=False, header=['file', 'gender', 'speaker', 'emotion', 'valence', 'arousal', 'dominance', 'naturalness'])

    train_list_df = pd.DataFrame(train_list).fillna(3)
    train_list_df.to_csv('improv_test.csv', index=False, header=['file', 'gender', 'speaker', 'emotion', 'valence', 'arousal', 'dominance', 'naturalness'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./MSP-IMPROV/', help='path to dataset')
    args = parser.parse_args()
    main(args.data_path)
