#!/usr/bin/env python3
"""process_database.py -- EMOVOME (text)

Converts the EMOVOME (Emotional Voice Messages) database [1] into
nkululeko CSV file lists for a **text-only** experiment: emovome_train.csv,
emovome_dev.csv, emovome_test.csv (the official, speaker-independent
partitions), plus emovome_all.csv with all three combined (a "split"
column tells them apart).

This needs only the openly downloadable metadata -- no request to the
corresponding author, no restricted-access audio. The actual signal
used for prediction is the manually corrected transcript of each voice
message (see exp.ini: FEATS.type = ['bert']), not audio content at
all. nkululeko still expects a readable audio file per row (to work
out each row's duration), so this script also writes a tiny placeholder
silent .wav per row -- its content is never read by the text feature
extractor, it exists purely so nkululeko's bookkeeping has a file to
point at.

Setup
-----
Download and unzip "emovome.zip" from the EMOVOME Zenodo record [2]
(1.3 GB, open access) into this directory -- see README.md for the
exact commands. It contains the metadata this script reads:
    labels.csv
    participants_ids.csv
    transcriptions.csv
    Baseline_emotion_recognition/partitions.csv
    Questionnaires/sociodemographic_english.csv

Usage
-----
Run with no options at all from inside the unzipped emovome folder:

    python process_database.py

Override --zenodo_dir/--out_dir if your layout differs.

References
----------
[1] Gomez-Zaragoza et al., "EMOVOME Database: Advancing Emotion
    Recognition in Speech Beyond Staged Scenarios" (2024).
    https://doi.org/10.48550/arXiv.2403.02167
[2] https://doi.org/10.5281/zenodo.10694370
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf

# partitions.csv calls the dev set "val"; nkululeko's split_strategy
# values are train/dev/test (see docs/source/ini_file.md)
SPLIT_MAP = {"train": "train", "val": "dev", "test": "test"}

# Silent placeholder audio: just long/loud enough that nothing downstream
# (e.g. a sanity VAD/duration check) mistakes it for a corrupt file.
_PLACEHOLDER_SR = 16000
_PLACEHOLDER_SECONDS = 0.5


def _write_placeholder_audio(out_dir, file_id):
    """A short silent .wav for file_id, so nkululeko has something to read
    a duration from. Its content carries no signal -- see module
    docstring -- so it is written once and reused across runs."""
    path = os.path.join(out_dir, f"{file_id}.wav")
    if not os.path.isfile(path):
        silence = np.zeros(int(_PLACEHOLDER_SR * _PLACEHOLDER_SECONDS), dtype="float32")
        sf.write(path, silence, _PLACEHOLDER_SR)
    return os.path.abspath(path)


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
        "--out_dir",
        default=".",
        help="Where to write the output CSVs and the placeholder_audio/ "
        "folder (default: current directory)",
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

    placeholder_dir = os.path.join(args.out_dir, "placeholder_audio")
    os.makedirs(placeholder_dir, exist_ok=True)
    df["file"] = df["file_id"].apply(
        lambda fid: _write_placeholder_audio(placeholder_dir, fid)
    )

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
