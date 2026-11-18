#!/usr/bin/env python3
"""process_database.py -- EMOVOME

Converts the EMOVOME (Emotional Voice Messages) database [1] into
nkululeko CSV file lists: emovome_train.csv, emovome_dev.csv,
emovome_test.csv (the official, speaker-independent partitions), plus
emovome_all.csv with all three combined (a "split" column tells them
apart).

Setup
-----
1. Download and unzip "emovome.zip" from the EMOVOME Zenodo record [2]
   (1.3 GB, open access -- no agreement needed for anything in it). It
   contains the metadata this script reads:
       labels.csv
       participants_ids.csv
       transcriptions.csv
       Baseline_emotion_recognition/partitions.csv
       Questionnaires/sociodemographic_english.csv
   --zenodo_dir below points at this unzipped folder.
2. The actual voice-message audio ("Emotions/<file_id>.ogg", one file
   per row of labels.csv) is NOT in that zip. Request it separately:
   sign "EMOVOME_agreement.pdf" (also in the zip) and send it to the
   corresponding author (jamarmo@htech.upv.es); they deliver the
   "Audios" folder directly. --audio_dir below points at its "Emotions"
   subfolder.

Usage
-----
Run with no options at all from inside the unzipped emovome folder
(see README.md for how to get there):

    python process_database.py

This reads the metadata from the current directory and, if
Audios/Emotions/ exists there and has files in it, the audio from
there too. Override either with --zenodo_dir/--audio_dir if your
layout differs, e.g.:

    python process_database.py \
        --zenodo_dir /path/to/unzipped/emovome \
        --audio_dir /path/to/Audios/Emotions

Without audio (i.e. before Audios/Emotions/ has real files in it) you
still get the CSVs -- useful to inspect the label distribution -- but
every "file" entry is empty, and the result is not runnable with
nkululeko.

References
----------
[1] Gomez-Zaragoza et al., "EMOVOME Database: Advancing Emotion
    Recognition in Speech Beyond Staged Scenarios" (2024).
    https://doi.org/10.48550/arXiv.2403.02167
[2] https://doi.org/10.5281/zenodo.10694370
"""

import argparse
import glob
import os
import sys

import pandas as pd

# partitions.csv calls the dev set "val"; nkululeko's split_strategy
# values are train/dev/test (see docs/source/ini_file.md)
SPLIT_MAP = {"train": "train", "val": "dev", "test": "test"}


def _find_audio(audio_dir, file_id):
    """Path to file_id's voice message, whatever its extension (the
    database ships Ogg Vorbis, but this also picks up a user's own
    converted copy, e.g. to .wav)."""
    matches = sorted(glob.glob(os.path.join(audio_dir, f"{file_id}.*")))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--zenodo_dir",
        default=".",
        help="Directory holding the unzipped emovome.zip contents "
        "(default: current directory)",
    )
    parser.add_argument(
        "--audio_dir",
        default=None,
        help="Directory with the voice-message audio, named "
        "<file_id>.<ext> (the restricted-access 'Emotions' folder; see "
        "module docstring). Default: <zenodo_dir>/Audios/Emotions; "
        "metadata-only CSVs are produced if that doesn't exist or is "
        "empty.",
    )
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Where to write the output CSVs (default: current directory)",
    )
    args = parser.parse_args()

    def _read(relpath):
        path = os.path.join(args.zenodo_dir, relpath)
        if not os.path.isfile(path):
            sys.exit(
                f"ERROR: {path} not found. --zenodo_dir must point at the "
                "unzipped emovome.zip (see this script's module docstring)."
            )
        return pd.read_csv(path)

    labels = _read("labels.csv")
    participants = _read("participants_ids.csv")
    transcriptions = _read("transcriptions.csv")
    partitions = _read(os.path.join("Baseline_emotion_recognition", "partitions.csv"))
    sociodemographic = _read(
        os.path.join("Questionnaires", "sociodemographic_english.csv")
    )

    df = labels[["file_id", "valence_C", "arousal_C", "category_E"]].rename(
        columns={
            "valence_C": "valence",
            "arousal_C": "arousal",
            "category_E": "category",
        }
    )
    df["category"] = df["category"].str.lower()

    df = df.merge(participants[["file_id", "participant"]], on="file_id", how="left")
    df = df.rename(columns={"participant": "speaker"})

    df = df.merge(
        transcriptions[["file_id", "Manual", "AWS"]], on="file_id", how="left"
    ).rename(columns={"Manual": "transcription", "AWS": "transcription_asr"})

    df = df.merge(partitions, on="file_id", how="left")
    unmapped = df["split"][~df["split"].isin(SPLIT_MAP)].unique()
    if len(unmapped):
        sys.exit(f"ERROR: unexpected partitions.csv split value(s): {unmapped}")
    df["split"] = df["split"].map(SPLIT_MAP)

    socio = sociodemographic[["participant", "gender", "age"]].rename(
        columns={"participant": "speaker", "age": "age_range"}
    )
    socio["gender"] = socio["gender"].str.lower()
    df = df.merge(socio, on="speaker", how="left")

    audio_dir = args.audio_dir or os.path.join(args.zenodo_dir, "Audios", "Emotions")
    has_audio = os.path.isdir(audio_dir) and any(os.scandir(audio_dir))
    if has_audio:
        df["file"] = df["file_id"].apply(lambda fid: _find_audio(audio_dir, fid))
        missing = df["file"].isna().sum()
        if missing:
            print(
                f"WARNING: {missing}/{len(df)} rows have no audio file under "
                f"{audio_dir} (expected <file_id>.<ext>) and are dropped.",
                file=sys.stderr,
            )
            df = df[df["file"].notna()].copy()
        df["file"] = df["file"].apply(os.path.abspath)
    else:
        print(
            f"WARNING: no audio found under {audio_dir}, 'file' column left "
            "empty -- the CSVs below are metadata-only, not runnable with "
            "nkululeko.",
            file=sys.stderr,
        )
        df["file"] = pd.NA

    df = df.drop(columns=["file_id"])
    column_order = [
        "file",
        "speaker",
        "gender",
        "age_range",
        "valence",
        "arousal",
        "category",
        "transcription",
        "transcription_asr",
        "split",
    ]
    df = df[column_order]

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "emovome_all.csv"), index=False)
    for split in ("train", "dev", "test"):
        split_df = df[df["split"] == split].drop(columns=["split"])
        out_path = os.path.join(args.out_dir, f"emovome_{split}.csv")
        split_df.to_csv(out_path, index=False)
        print(f"{split}: {len(split_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
