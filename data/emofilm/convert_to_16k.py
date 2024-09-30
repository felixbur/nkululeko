# python code to convert wav fils from 44.1kHz to 16kHz
# arguments: input_dir, output_dir
# sox must be installed

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, default='./EmoFilm/wav_corpus')
parser.add_argument('-o', '--output_dir', type=str, default='./EmoFilm/wav_corpus_16k')
args = parser.parse_args()

source_dir = args.input_dir
target_dir = args.output_dir

# create the target directory if it does not exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Define the target sample rate
target_sr = 16000

# Loop over all audio files in the source directory
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(('.wav', '.mp3')):
            # print(file)
            print(f"Resampling {os.path.join(root, file)} to 16kHz")

            # Define the input and output file paths
            input_path = os.path.join(root, file)
            # obtain the basename
            basename = os.path.basename(input_path)
            output_path = os.path.join(target_dir, basename[:-4] + '_16k.wav')
            
            # Use sox to resample the audio file
            subprocess.run(['sox', input_path, '-r', str(target_sr), output_path, 'gain', '1'])