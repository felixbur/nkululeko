# process_database.py -> CMU-MOSEI database
# input file:
# - label_paths: CMU_MOSEI_Labels.csv
# - db_paths: CMU-MOSEI

import argparse
import os

import pandas as pd


def preprocess(data_dirs, name, split_df, out_path):
    meta_data = []
    for ix, row in split_df.iterrows():
        # get wav file path
        # print(row['file'])
        filename = row['file'] + '_' + str(row['index']) + '.wav'
        file = os.path.join(data_dirs, 'Audio', 'Segmented_Audio',
                            name, filename)
        
        sentiment = str(row['label2a'])
        for r in (("0", "negative"), ("1", "positive")):
            sentiment = sentiment.replace(*r)
        
        emotion = str(row['label6'])
        for r in (("0", "hap"), ("1", "sad"), ("2", "ang"), ("3", "sur"), ("4", "dis"), ("5", "fea")):
            emotion = emotion.replace(*r)
        
        meta_data.append({
                'file': file,
                'sentiment': sentiment,
                'emotion': emotion,
                })
        
        # write to csv
    meta_data_df = pd.DataFrame(meta_data)
    meta_data_df.to_csv(out_path, index=False)    
    print(f'Wrote {name} partition with {len(meta_data)} samples to {out_path}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./CMU-MOSEI',
                        help='Path to CMU-MOSEI directory.')
    args = parser.parse_args()
    label_path = 'CMU_MOSEI_Labels.csv'
    data = pd.read_csv(label_path)
    for i, split_name in enumerate(['train', 'dev', 'test']):
        print(f'Processing {split_name} (split == {i}).')
        split_df = data[data['split'] == i]
        preprocess(args.data_dir, split_name, split_df, 
                   f'mosei_{split_name}.csv')

    