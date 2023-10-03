# process_database.py
# process IEMOCAP database to match the format of Nkululeko
# adopted from S3PRL emotion downstream task

import argparse
import glob
import os
import re
from os.path import basename
from os.path import join as path_join
from os.path import splitext

import pandas as pd

LABEL_DIR_PATH = 'dialog/EmoEvaluation'
WAV_DIR_PATH = 'sentences/wav'


def get_wav_paths(data_dirs):
    wav_paths = glob.glob(f"{data_dirs}/**/*.wav", recursive=True)
    wav_dict = {}
    for wav_path in wav_paths:
        wav_name = splitext(basename(wav_path))[0]
        start = wav_path.find('Session')
        wav_path = wav_path[start:]
        wav_dict[wav_name] = wav_path

    return wav_dict


def preprocess(data_dirs, paths, out_path):
    meta_data = []
    for path in paths:
        wav_paths = get_wav_paths(path_join(data_dirs, path, WAV_DIR_PATH))
        label_dir = path_join(data_dirs, path, LABEL_DIR_PATH)
        label_paths = list(os.listdir(label_dir))
        label_paths = [label_path for label_path in label_paths
                       if splitext(label_path)[1] == '.txt']
        for label_path in label_paths:
            with open(path_join(label_dir, label_path)) as f:
                for line in f:
                    if line[0] != '[':
                        continue
                    line = re.split('[\t\n]', line)
                    line = list(filter(None, line))
                    if line[2] not in ['neu', 'hap', 'ang', 'sad', 'exc']:
                        continue
                    if line[1] not in wav_paths:
                        continue
                    meta_data.append({
                        'file': os.path.join(data_dirs, wav_paths[line[1]]),
                        'emotion': line[2].replace('exc', 'hap'),
                        'speaker': re.split('_', basename(wav_paths[line[1]]))[0],
                        'gender': re.split('_', basename(wav_paths[line[1]]))[-1][0]
                    })
    data = {
        'labels': {'neu': 0, 'hap': 1, 'ang': 2, 'sad': 3},
        'meta_data': meta_data
    }

    # print number of utterances
    print(f"Total number of {os.path.basename(out_path)}: {len(data['meta_data'])}")

    # save to CSV file
    pd.DataFrame(data['meta_data']).to_csv(out_path, index=False)


def main(data_dir):
    """Main function."""
    paths = list(os.listdir(data_dir))
    paths = [path for path in paths if path[:7] == 'Session']
    paths.sort()
    out_dir = os.path.dirname(__file__)
    # make Session 1-3 as training set, Sesion4 for dev, Session 5 as test set
    preprocess(data_dir, ['Session5'], path_join(out_dir, 'iemocap_test.csv'))
    preprocess(data_dir, ['Session4'], path_join(out_dir, 'iemocap_dev.csv'))
    preprocess(data_dir, ['Session1', 'Session2', 'Session3'], path_join(out_dir, 'iemocap_train.csv'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./IEMOCAP_full_release')
    args = parser.parse_args()
    main(args.data_dir)
