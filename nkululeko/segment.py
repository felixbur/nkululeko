# segment.py
# segment a database

from nkululeko.experiment import Experiment
import configparser
from nkululeko.util import Util
from  nkululeko.constants import VERSION
import argparse
import os
import torch
import audformat
from audformat.utils import to_filewise_index
from audformat import segmented_index
import pandas as pd 

# initialize the VAD model
SAMPLING_RATE = 16000
torch.set_num_threads(1)
vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False)
(get_speech_timestamps,
save_audio,
read_audio,
VADIterator,
collect_chunks) = vad_utils

def main(src_dir):
    parser = argparse.ArgumentParser(description='Call the nkululeko framework.')
    parser.add_argument('--config', default='exp.ini', help='The base configuration')
    args = parser.parse_args()
    if args.config is not None:
        config_file = args.config    
    else:
        config_file = f'{src_dir}/exp.ini'

    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print(f'ERROR: no such file: {config_file}')
        exit()
        
    # load one configuration per experiment
    config = configparser.ConfigParser()
    config.read(config_file)
    # create a new experiment
    expr = Experiment(config)
    util = Util('segment')
    util.debug(f'running {expr.name}, nkululeko version {VERSION}')

    # load the data
    expr.load_datasets()

    # split into train and test
    expr.fill_train_and_tests()
    util.debug(f'train shape : {expr.df_train.shape}, test shape:{expr.df_test.shape}')

    # segment
    segment_target = util.config_val('DATA', 'segment_target', '_seg')
    # this if a specific dataset is to be segmented
    # segment_db = util.config_val('DATA', 'segment', False)
    # if segment_db:
    #     for dataset in expr.datasets.keys:
    #         if segment_db == dataset:
    #             df = expr.datasets[dataset].df
    #             util.debug(f'segmenting {dataset}') 
    #             df_seg = segment_dataframe(df) 
    #             name = f'{dataset}{segment_target}'
    #             df_seg.to_csv(f'{expr.data_dir}/{name}.csv')

    sample_selection = util.config_val('DATA', 'segment', 'all')
    if sample_selection=='all':
        df = pd.concat([expr.df_train, expr.df_test])
    elif sample_selection=='train':
        df = expr.df_train
    elif sample_selection=='test':
        df = expr.df_test
    else:
        util.error(f'unknown segmentation selection specifier {sample_selection}, should be [all | train | test]')

    if "duration" not in df.columns:
        df = df.drop(columns=['duration'], inplace=True)
    util.debug(f'segmenting train and test set: {df.shape[0]} samples') 
    df_seg = segment_dataframe(df) 
    def calc_dur(x):
        starts = x[1]
        ends = x[2]
        return (ends - starts).total_seconds()
    df_seg['duration'] = df_seg.index.to_series().map(lambda x:calc_dur(x)) 
    dataname = '_'.join(expr.datasets.keys())
    name = f'{dataname}{segment_target}'
    df_seg.to_csv(f'{expr.data_dir}/{name}.csv')
    print('')
    util.debug(f'saved {name}.csv to {expr.data_dir}, {df_seg.shape[0]} samples')
    print('DONE')




def get_segmentation(file):
#    print(f'segmenting {file[0]}')
    print('.', end='')
    wav = read_audio(file[0], sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLING_RATE)
    files, starts, ends = [], [], []
    for entry in speech_timestamps:
        start = float(entry['start']/1000.0)
        end = float(entry['end']/1000.0)
        files.append(file[0])
        starts.append(start)
        ends.append(end)
    seg_index = segmented_index(files, starts, ends)
    return seg_index

def segment_dataframe(df):
    dfs = []
    for file, values in df.iterrows():
        index = get_segmentation(file)
        dfs.append(
            pd.DataFrame(
                values.to_dict(),
                index,
            )
        )
    return audformat.utils.concat(dfs)


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.abspath(__file__))
    main(cwd) # use this if you want to state the config file path on command line