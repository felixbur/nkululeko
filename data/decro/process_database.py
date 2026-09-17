#!/usr/bin/env python3
"""
Process the DECRO (DEepfake CROss-lingual evaluation) dataset and generate
CSV files for nkululeko.

DECRO ships as 6 splits, each with a protocol file and matching audio
directory: {ch,en}_{train,dev,eval}.{txt,/}. Protocol lines are ASVspoof-
style, whitespace-separated:

    SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY

where KEY is 'bonafide' or 'spoof' (mapped here to nkululeko's 'real'/
'fake' convention, matching every other dataset in this project), and
AUDIO_FILE_NAME has no extension (the wav lives at <split>/<name>.wav).

See https://github.com/petrichorwq/DECRO-dataset for the paper this
dataset accompanies (Ba et al., "Transferring Audio Deepfake Detection
Capability across Languages", WWW 2023) -- it's designed specifically to
evaluate cross-lingual generalization: the English and Chinese subsets
use matched spoofing algorithms (HiFiGAN, Multiband-MelGAN, PWG, Tacotron,
FastSpeech2, VITS, Starganv2-vc, ...) from independent bona-fide corpora,
so the language axis is isolated from the spoofing-method axis (unlike
CVoiceFake-small, where all languages share one TTS pipeline).

Output CSV files (columns: file, label, language, system, speaker):
- decro_en_train.csv, decro_en_dev.csv, decro_en_eval.csv
- decro_ch_train.csv, decro_ch_dev.csv, decro_ch_eval.csv
- decro_en.csv, decro_ch.csv (each language's 3 splits combined)
- decro.csv (everything combined)
"""

import argparse
from pathlib import Path

import pandas as pd

SPLITS = ["train", "dev", "eval"]
LANGUAGES = ["en", "ch"]
KEY_MAP = {"bonafide": "real", "spoof": "fake"}


def read_protocol(protocol_path, audio_dir, language):
    rows = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id, audio_name, _, system_id, key = parts[:5]
            label = KEY_MAP.get(key)
            if label is None:
                continue
            rows.append(
                {
                    "file": str(audio_dir / f"{audio_name}.wav"),
                    "label": label,
                    "language": language,
                    "system": system_id,
                    "speaker": speaker_id,
                }
            )
    return pd.DataFrame(rows)


def main(data_dir, output_dir):
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: directory not found: {data_dir}")
        return

    per_language = {lang: [] for lang in LANGUAGES}
    all_data = []

    for lang in LANGUAGES:
        for split in SPLITS:
            protocol_path = data_dir / f"{lang}_{split}.txt"
            audio_dir = data_dir / f"{lang}_{split}"
            if not protocol_path.exists() or not audio_dir.exists():
                print(f"WARNING: missing {protocol_path} or {audio_dir}, skipping")
                continue

            df = read_protocol(protocol_path, audio_dir, lang)
            if df.empty:
                print(f"WARNING: no rows parsed from {protocol_path}")
                continue

            csv_path = output_dir / f"decro_{lang}_{split}.csv"
            df.to_csv(csv_path, index=False)
            counts = df["label"].value_counts().to_dict()
            print(f"{lang}_{split}: {len(df)} samples -> {csv_path}  {counts}")

            per_language[lang].append(df)
            all_data.append(df)

    for lang in LANGUAGES:
        if per_language[lang]:
            combined = pd.concat(per_language[lang], ignore_index=True)
            csv_path = output_dir / f"decro_{lang}.csv"
            combined.to_csv(csv_path, index=False)
            print(f"{lang} combined: {len(combined)} samples -> {csv_path}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        csv_path = output_dir / "decro.csv"
        combined.to_csv(csv_path, index=False)
        print(f"\nDONE: {len(combined)} total samples -> {csv_path}")
    else:
        print("\nERROR: no data processed. Check dataset structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process the DECRO cross-lingual deepfake dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/bagus/data/DECRO_16k",
        help=(
            "Path to the DECRO dataset directory (contains "
            "{en,ch}_{train,dev,eval}.txt/) -- use the 16kHz-resampled copy "
            "from resample_to_16k.py, not the raw extracted archive: "
            "nkululeko's wav2vec2 extractor requires exactly 16kHz and "
            "doesn't resample, and ~43% of DECRO's raw audio ships at "
            "other rates (22050/24000/21900/44100 Hz)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Path to the output directory for CSV files",
    )
    args = parser.parse_args()

    main(args.data_dir, args.output_dir)
