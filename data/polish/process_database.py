# process_database.py for POLISH dataset

import argparse
from pathlib import Path

import pandas as pd

# emotion mapping , a: anger, f: fear, n:n neutral
emotion_mapping = {
    "a": "anger",
    "s": "fear",
    "n": "neutral",
}


def process_database(data_dir, output_dir):
    # check if the data directory exists
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    # check if the output directory exists, create if not
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # read the database
    wavs = list(data_dir.glob("*.wav"))
    df = pd.DataFrame(wavs, columns=["file"])

    # get filename path with extension
    df["file"] = df["file"].apply(lambda x: x.name)

    # get emotion, speaker, and gender from filename
    # file format <speaker>.<emotion>_<statement>_<repition>.wav
    df["speaker"] = df["file"].apply(lambda x: x.split(".")[0])

    # get gender from first letter of speaker, m for male, f for female
    df["gender"] = df["speaker"].apply(
        lambda x: "male" if x[0] == "m" else (
            "female" if x[0] == "f" else "unknown")
    )

    # prepend the speaker with the dataset name
    df["speaker"] = "POLISH_" + df["speaker"]

    # get emotion from emotion mapping
    df["emotion"] = df["file"].apply(
        lambda x: emotion_mapping[x.split(".")[1][0]])

    # add language
    df["language"] = "polish"

    # prepend data_dir to file
    df["file"] = df["file"].apply(lambda x: str(data_dir / x))

    # allocate speaker 1, 2, 3 for train, 4 for dev, 5 for test
    speakers = df["speaker"].unique()
    train_speakers, dev_speakers, test_speakers = (
        speakers[:3],
        speakers[3:4],
        speakers[4:],
    )

    # loop over train, dev, and test, save as csv and print length
    for set_name in ["train", "dev", "test"]:
        df_set = df[df["speaker"].isin(eval(f"{set_name}_speakers"))]
        df_set.to_csv(output_dir / f"polish_{set_name}.csv", index=False)
        print(
            f"Saved {len(df_set)} samples to {output_dir / f'polish_{set_name}.csv'}")
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process POLISH database")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="POLISH",
        help="Directory containing the POLISH data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the processed data",
    )
    args = parser.parse_args()
    process_database(args.data_dir, args.output_dir)
