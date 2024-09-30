import argparse
import glob
import os
from os.path import basename
from os.path import join as path_join
from os.path import splitext

import pandas as pd

LABEL_DIR_PATH = 'Labels'
WAV_DIR_PATH = 'Audios'


def get_wav_paths_podcast(paths):
    wav_paths = glob.glob(f"{paths}/**/*.wav", recursive=True)
    wav_dict = {}
    for wav_path in wav_paths:
        wav_name = splitext(basename(wav_path))[0]
        wav_dict[wav_name] = wav_path
    return wav_dict

def preprocess(data_dirs, paths, wav_paths, out_path):
    meta_data = []
    for idx, row in paths.iterrows():
        if row['EmoClass'] not in ['A', 'H', 'N', 'S']:
            continue
        # remove outlier (stereo, 44.1 kHz)
        if row['FileName'] == 'MSP-PODCAST_1023_0235.wav':
            continue
        filename = row['FileName'][:-4]
        label = row['EmoClass']
        valence = row['EmoVal']
        arousal = row['EmoAct']
        dominance = row['EmoDom']
        #speaker = row['SpkrID']
        #gender = row['Gender']
        for r in (('A', 'ang'), ('H', 'hap'), 
                ('N', 'neu'), ('S', 'sad')):
            label = label.replace(*r)
        meta_data.append({
                    'file': wav_paths[filename],
                    'emotion': label,
                    'valence': valence,
                    'arousal': arousal,
                    'dominance': dominance,
                    #'speaker': speaker,
                    #'gender': gender
                    })

    # write to csv
    df = pd.DataFrame(meta_data)
    df.to_csv(out_path, index=False)


def main(data_dir):
    """Main function."""
    out_dir = os.path.dirname(__file__)
    wav_paths = get_wav_paths_podcast(path_join(f"{data_dir}/{WAV_DIR_PATH}"))
    data = pd.read_csv(path_join(f"{data_dir}/{LABEL_DIR_PATH}",
                                       'labels_concensus.csv'))
    # Train and test1 1 are used for test
    train_path = data.loc[
        (data.Split_Set == 'Train')]
    # Development/validation set
    dev_path = data.loc[
        (data['Split_Set'] == 'Validation')]
    # Using test1 as test
    test_path1 = data.loc[
        (data['Split_Set'] == 'Test1')]  
    test_path2 = data.loc[
        (data['Split_Set'] == 'Test2')]
    
    preprocess(data_dir, train_path, wav_paths, 
               path_join(f"{out_dir}", 'podcast_train.csv'))
    preprocess(data_dir, dev_path, wav_paths, path_join(f"{out_dir}",
                                                         'podcast_dev.csv'))
    preprocess(data_dir, test_path1, wav_paths, 
               path_join(f"{out_dir}", 'podcast_test1.csv'))
    preprocess(data_dir, test_path2, wav_paths,
                path_join(f"{out_dir}", 'podcast_test2.csv'))


if __name__ == "__main__":
    """input the argument when calling this file
    e.g. python PODCAST_preprocess /path/to/PODCAST"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./MSP-PODCAST-Publish-1.8', help='Path to the dataset')
    args = parser.parse_args()
    main(args.data_dir)
