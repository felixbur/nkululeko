import argparse
import os

import pandas as pd

# CVoiceFake-small: Common Voice clips (Bonafide) plus five vocoder/synthesis
# systems producing spoofed counterparts, split by language.
LANGUAGES = ["de", "en", "fr", "it", "zh-CN"]
REAL_SYSTEM = "Bonafide"
FAKE_SYSTEMS = [
    "griffin_lim_generated",
    "vctk_multi_band_melgan.v2_generated",
    "vctk_parallel_wavegan.v1_generated",
    "vctk_style_melgan.v1_generated",
    "world_generated",
]


def main(data_dir, output_dir):
    """Process the CVoiceFake-small dataset into nkululeko train/dev/test CSVs.

    Expects the folder layout shipped by CVoiceFake-small: one subfolder per
    language, each containing Bonafide/ (real Common Voice clips) plus one
    subfolder per vocoder/synthesis system (fake clips), all as .mp3 files.
    The audio stays where it is (data_dir); only the generated CSVs are
    written into the nkululeko repo at output_dir, so the large audio corpus
    itself never needs to be copied into or committed to the repo.

    No speaker manifest (e.g. Common Voice's client_id) ships with this
    dataset -- filenames only carry the original Common Voice clip id, which
    is not a speaker id. Every file is therefore treated as its own
    pseudo-speaker, so the speaker-grouped split below degenerates to a
    stratified random file-level split. If a truly speaker-independent split
    is needed, fetch the matching Common Voice validated.tsv separately and
    join on clip id to recover real client_id speaker groups.
    """
    rows = []
    for lang in LANGUAGES:
        lang_dir = os.path.join(data_dir, lang)
        if not os.path.isdir(lang_dir):
            print(f"WARNING: {lang_dir} not found, skipping")
            continue
        for system in [REAL_SYSTEM] + FAKE_SYSTEMS:
            sys_dir = os.path.join(lang_dir, system)
            if not os.path.isdir(sys_dir):
                print(f"WARNING: {sys_dir} not found, skipping")
                continue
            label = "real" if system == REAL_SYSTEM else "fake"
            for fname in sorted(os.listdir(sys_dir)):
                if not fname.lower().endswith(".mp3"):
                    continue
                rows.append(
                    {
                        "file": os.path.join(sys_dir, fname),
                        "speaker": f"{lang}_{system}_{fname}",
                        "label": label,
                        "language": lang,
                        "system": system,
                    }
                )

    df = pd.DataFrame(rows)

    # "Speaker"-independent split (60% train, 20% dev, 20% test). Since every
    # group above is a singleton, this is equivalent to a plain random
    # file-level split, kept identical in structure to the other nkululeko
    # deepfake dataset scripts for consistency.
    speakers = df["speaker"].unique()
    speakers_shuffled = pd.Series(speakers).sample(frac=1, random_state=42).values
    total_speakers = len(speakers_shuffled)

    train_end = int(0.6 * total_speakers)
    dev_end = int(0.8 * total_speakers)

    train_speakers = speakers_shuffled[:train_end]
    dev_speakers = speakers_shuffled[train_end:dev_end]
    test_speakers = speakers_shuffled[dev_end:]

    df_train = df[df["speaker"].isin(train_speakers)].reset_index(drop=True)
    df_dev = df[df["speaker"].isin(dev_speakers)].reset_index(drop=True)
    df_test = df[df["speaker"].isin(test_speakers)].reset_index(drop=True)
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_file = os.path.join(output_dir, "cvoicefake_train.csv")
    dev_file = os.path.join(output_dir, "cvoicefake_dev.csv")
    test_file = os.path.join(output_dir, "cvoicefake_test.csv")
    all_file = os.path.join(output_dir, "cvoicefake.csv")

    df_train.to_csv(train_file, index=False)
    df_dev.to_csv(dev_file, index=False)
    df_test.to_csv(test_file, index=False)
    df_shuffled.to_csv(all_file, index=False)

    print(f"Created {train_file} with {len(df_train)} samples")
    print(f"Created {dev_file} with {len(df_dev)} samples")
    print(f"Created {test_file} with {len(df_test)} samples")
    print(f"Created {all_file} with {len(df_shuffled)} samples")

    print("\nLabel distribution in complete dataset:")
    print(df["label"].value_counts())
    print("\nLanguage distribution in complete dataset:")
    print(df["language"].value_counts())
    print("\nSystem distribution in complete dataset:")
    print(df["system"].value_counts())

    print("\nLabel distribution in train set:")
    print(df_train["label"].value_counts())
    print("\nLabel distribution in dev set:")
    print(df_dev["label"].value_counts())
    print("\nLabel distribution in test set:")
    print(df_test["label"].value_counts())
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CVoiceFake-small audio deepfake dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/data/CVoiceFake_small"),
        help="Path to the extracted CVoiceFake-small dataset directory (audio stays here)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write the generated CSVs into (default: this script's dir)",
    )
    args = parser.parse_args()

    main(args.data_dir, args.output_dir)
