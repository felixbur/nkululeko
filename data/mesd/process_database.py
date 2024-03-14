#!/usr/bin/env python3
# process_database.py --> MSED

import argparse
from email import parser
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

gender_map = {
    "F": "female",
    "M": "male",
    "C": "child",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='MESD/Mexican Emotional Speech Database (MESD)', help='Path to the database file')
    parser.add_argument('--output_dir', type=str, default='.', help='Path to the output directory')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir()
        
    paths = list(data_dir.glob('**/*.wav'))
    emotions = [p.stem.split("_")[0].lower() for p in paths]
    voices = [p.stem.split("_")[1] for p in paths]
    genders = [gender_map[p.stem.split("_")[1]] for p in paths]
    languages = ["spanish" for p in paths]
    corpus = [p.stem.split("_")[2] for p in paths]
    
    if str(data_dir).find("reduced") != -1:
        reduced_level = [p.stem.split("_")[3] for p in paths]
        words = [" ".join(p.stem.split("_")[4:]) for p in paths]
        df = pd.DataFrame({"file": paths, "emotion": emotions, "gender": genders, "language": languages, "reduced_level": reduced_level, "word": words})       

    else:
        words = [" ".join(p.stem.split("_")[3:]) for p in paths]
        df = pd.DataFrame({"file": paths, "emotion": emotions, "gender": genders, "language": languages, "word": words})

    # split train and test
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["emotion"])
    
    # save to csv
    df.to_csv(output_dir / "mesd.csv", index=False)
    df_train.to_csv(output_dir / "mesd_train.csv", index=False)
    df_test.to_csv(output_dir / "mesd_test.csv", index=False)

    print(f"Total: {len(df)}, Train: {len(df_train)}, Test: {len(df_test)}")


if __name__ == "__main__":
    main()
