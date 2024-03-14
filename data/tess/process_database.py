# process_database.py --> TESS

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Process TESS database')
parser.add_argument('--data_dir', type=str, default='./TESS',
                    help='path to the data directory')
args = parser.parse_args()

data_dir = Path(args.data_dir)
assert data_dir.exists()

# emotion map
emotion_map = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "ps": "surprise",
    "sad": "sadness",
    "neutral": "neutral",
}

def main():
    input_dir = Path(args.data_dir)
    paths = list(input_dir.glob('**/*.wav'))
    if len(paths) == 0:
        raise FileNotFoundError("No audio files found.")
    
    # file = [p.name for p in paths]  # basename only
    file = [p for p in paths]  # full path
    emotion = [emotion_map[p.stem.split('_')[-1]] for p in paths]
    speaker = [p.stem[:3] for p in paths]

    # write to csv
    df = pd.DataFrame(data={'file': file, 'emotion': emotion, 'speaker': speaker})

    # split to train, dev, test sets using scikit-learn
    # based on distribution of emotions (stratified) -> 80% each emottion for train
    
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # save to csv
    train.to_csv('tess_train.csv', index=False)
    test.to_csv('tess_test.csv', index=False)

if __name__ == '__main__':
    main()