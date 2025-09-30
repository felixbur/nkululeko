#!/usr/bin/env python3
"""
Process EATD dataset (ICASSP 2022 Depression).
- Splits train/ participants into train + dev (speaker independent).
- Uses validation/ participants as test set.
- Only includes *_out.wav files (cleaned audio).
- Generates CSVs: eatd_train.csv, eatd_dev.csv, eatd_test.csv
"""

import csv
from pathlib import Path
from sklearn.model_selection import train_test_split

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = Path("./data/eatd/EATD")   # Root containing train/ and validation/
OUTPUT_DIR = Path("./data/eatd")
SDS_THRESHOLD = 53.0

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def get_participant_data(participant_folder):
    """Return list of (file_path, speaker, emotion, task, sds) for one participant."""
    label_path = participant_folder / "new_label.txt"
    if not label_path.exists():
        print(f"⚠️ Missing new_label.txt in {participant_folder}, skipping...")
        return []

    try:
        # some files might have spaces/newlines → take first number only
        sds = float(label_path.read_text().strip().split()[0])
    except ValueError:
        print(f"⚠️ Could not parse SDS in {label_path}, skipping...")
        return []

    task = "depressed" if sds >= SDS_THRESHOLD else "non-depressed"
    speaker = participant_folder.name

    rows = []
    for emotion in ["negative", "neutral", "positive"]:
        wav_file = participant_folder / f"{emotion}_out.wav"
        if wav_file.exists():
            rows.append((
                str(wav_file.resolve()),  # absolute path
                speaker,
                emotion,
                task,
                sds
            ))
        else:
            print(f"ℹ️ Missing file {wav_file}, skipping...")
    return rows


# -------------------------
# MAIN PROCESS
# -------------------------
print("🔎 Scanning dataset...")

train_rows, val_rows = [], []

# collect separately from train/ and validation/
for subset in ["train", "validation"]:
    subset_path = DATA_DIR / subset
    if not subset_path.exists():
        print(f"⚠️ Warning: {subset_path} not found, skipping...")
        continue

    for participant_folder in subset_path.iterdir():
        if participant_folder.is_dir():
            rows = get_participant_data(participant_folder)
            if subset == "train":
                train_rows.extend(rows)
            else:
                val_rows.extend(rows)

print(f"✅ Found {len(train_rows)} audio files in train/")
print(f"✅ Found {len(val_rows)} audio files in validation/")

if len(train_rows) == 0 or len(val_rows) == 0:
    print("❌ ERROR: No audio files found. Check DATA_DIR path and file structure.")
    exit(1)

# split only train rows by speaker
speakers = sorted(list({row[1] for row in train_rows}))
print(f"👥 Total train speakers: {len(speakers)}")

train_speakers, dev_speakers = train_test_split(speakers, test_size=0.2, random_state=42)

print(f"📊 Train speakers: {len(train_speakers)}")
print(f"📊 Dev speakers: {len(dev_speakers)}")
print(f"📊 Test speakers: {len({row[1] for row in val_rows})}")

# create OUTPUT_DIR if not exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# save CSVs
splits = {
    "train": [r for r in train_rows if r[1] in train_speakers],
    "dev":   [r for r in train_rows if r[1] in dev_speakers],
    "test":  val_rows,
}

for set_name, rows_set in splits.items():
    csv_path = OUTPUT_DIR / f"eatd_{set_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "speaker", "emotion", "task", "sds"])
        writer.writerows(rows_set)
    print(f"💾 Saved {len(rows_set)} samples to {csv_path}")

print("✅ DONE")
