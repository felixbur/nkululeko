# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "audformat>=1.0",
#     "matplotlib>=3.9",
#     "pandas>=2.0",
#     "seaborn>=0.13",
# ]
# ///
"""
Build an audformat database from the StressDat metadata CSVs.

Requires stressdat_files.csv and stressdat_segments.csv produced by
01_generate_tables.py.  Saves an audformat database to <repo-root>/build/.

Tables:
  files            – filewise, one row per utterance
  segments.train   – segmented VAD segments, train speakers
  segments.dev     – segmented VAD segments, dev speakers
  segments.test    – segmented VAD segments, test speakers

Columns (all tables):
  induced_stress – intended stress level (0=neutral, 1=medium, 2=high)
  gender         – speaker gender (female / male)
  stress_level   – mean perceptual stress score across annotators (0–100)
  duration       – duration in seconds
  speaker        – speaker identifier (e.g. s06h)

Fixed speaker-disjunct train/dev/test split.
"""

import shutil
from pathlib import Path

import audformat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parent
CSV_PATH  = ROOT / "stressdat_files.csv"
SEG_PATH  = ROOT / "stressdat_segments.csv"
DB_PATH   = ROOT / "build"

df     = pd.read_csv(CSV_PATH, index_col="file_path")
seg_df = pd.read_csv(SEG_PATH)

print("intializing db")

db = audformat.Database(
    name="stressdat",
    source="https://www.sav.sk/journals/uploads/11080936579-589-stressdat-database-of-speech-under-stress-in-slovak.pdf",
    usage=audformat.define.Usage.RESEARCH,
    languages="sk",
    description=(
        "Slovak speech stress database (StressDat). "
        "29 Speakers recorded under calming, neutral, medium, and high-stress conditions."
        "Annotated by 5 listeners."
    ),
    author="Miroslava Rajčáni et al.",
    organization="Slovak Academy of Sciences",
    license="custom",
)

db.schemes["induced_stress"] = audformat.Scheme(
    dtype=audformat.define.DataType.INTEGER,
    labels=[0, 1, 2],
    description="Intended stress level from recording scenario (0=neutral, 1=medium, 2=high)",
)

db.schemes["gender"] = audformat.Scheme(
    dtype=audformat.define.DataType.STRING,
    labels=["female", "male"],
    description="Speaker gender",
)

db.schemes["stress_level"] = audformat.Scheme(
    dtype=audformat.define.DataType.FLOAT,
    minimum=0.0,
    maximum=100.0,
    description="Mean perceptual stress score assigned by annotators",
)

db.schemes["duration"] = audformat.Scheme(
    dtype=audformat.define.DataType.FLOAT,
    minimum=0.0,
    description="Duration in seconds (full file for 'files' table, speech segment for 'segments' table)",
)

# -- speaker split -----------------------------------------------------------
# Defined using splitutils to balance stress_level and speaker sex across splits,
# while keeping speakers disjoint.

TRAIN_SPEAKERS = ['s02a', 's03a', 's04h', 's05h', 's06h', 's07n', 's08h', 's10h',
                  's12h', 's13h', 's14h', 's16h', 's17h', 's19h', 's21h', 's22h',
                  's24h', 's25h', 's26h', 's27h']
DEV_SPEAKERS   = ['s01a', 's11h', 's18h']
TEST_SPEAKERS  = ['s09h', 's15h', 's20h', 's23h', 's28h', 's29h']

db.schemes["speaker"] = audformat.Scheme(
    dtype=audformat.define.DataType.STRING,
    labels=sorted(TRAIN_SPEAKERS + DEV_SPEAKERS + TEST_SPEAKERS),
    description="Speaker identifier",
)

# -- filewise table ----------------------------------------------------------
file_index = audformat.filewise_index(df.index.tolist())

db["files"] = audformat.Table(file_index, description="Per-utterance metadata")
db["files"]["induced_stress"] = audformat.Column(scheme_id="induced_stress")
db["files"]["gender"]         = audformat.Column(scheme_id="gender")
db["files"]["stress_level"]   = audformat.Column(scheme_id="stress_level")
db["files"]["duration"]       = audformat.Column(scheme_id="duration")
db["files"]["speaker"]        = audformat.Column(scheme_id="speaker")

db["files"].set(
    {
        "induced_stress": df["induced_stress"].tolist(),
        "gender":         df["gender"].tolist(),
        "stress_level":   df["stress_level"].tolist(),
        "duration":       df["duration"].tolist(),
        "speaker":        df["speaker"].tolist(),
    }
)


speaker_split: dict[str, str] = (
    {s: "train" for s in TRAIN_SPEAKERS}
    | {s: "dev"   for s in DEV_SPEAKERS}
    | {s: "test"  for s in TEST_SPEAKERS}
)

speaker_ids = np.array([Path(p).parts[1] for p in df.index])

db.splits["train"] = audformat.Split(type=audformat.define.SplitType.TRAIN)
db.splits["dev"]   = audformat.Split(type=audformat.define.SplitType.DEVELOP)
db.splits["test"]  = audformat.Split(type=audformat.define.SplitType.TEST)

seg_speaker = np.array([Path(f).parts[1] for f in seg_df["file"]])
seg_df["split"] = pd.array([speaker_split[s] for s in seg_speaker], dtype="string")

for split in ("train", "dev", "test"):
    mask   = seg_df["split"] == split
    subset = seg_df[mask]
    idx    = audformat.segmented_index(
        files=subset["file"].tolist(),
        starts=subset["start"].tolist(),
        ends=subset["end"].tolist(),
    )
    table_id = f"segments.{split}"
    db[table_id] = audformat.Table(
        idx,
        split_id=split,
        description=f"VAD speech segments — {split} set",
    )
    db[table_id]["induced_stress"] = audformat.Column(scheme_id="induced_stress")
    db[table_id]["gender"]         = audformat.Column(scheme_id="gender")
    db[table_id]["stress_level"]   = audformat.Column(scheme_id="stress_level")
    db[table_id]["duration"]       = audformat.Column(scheme_id="duration")
    db[table_id]["speaker"]        = audformat.Column(scheme_id="speaker")
    db[table_id].set({
        "induced_stress": subset["induced_stress"].tolist(),
        "gender":         subset["gender"].tolist(),
        "stress_level":   subset["stress_level"].tolist(),
        "duration":       subset["duration"].tolist(),
        "speaker":        subset["speaker"].tolist(),
    })

REPO = Path(__file__).parent / "repo"

DB_PATH.mkdir(exist_ok=True)
db.save(DB_PATH, storage_format="csv")

# -- copy audio files --------------------------------------------------------

AUDIO_DST = DB_PATH / "audio"
AUDIO_DST.mkdir(parents=True, exist_ok=True)
all_files = df.index.tolist()
print(f"\nCopying {len(all_files):,} audio files to {AUDIO_DST} …")
for stored_path in all_files:
    bare = stored_path[len("audio/"):]
    src = REPO / bare
    dst = AUDIO_DST / bare
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
print("Audio copy done.")

counts = seg_df["split"].value_counts()
spk_counts = pd.Series(speaker_split).value_counts()
print(f"Saved audformat database ({len(df):,} files, {len(seg_df):,} segments) → {DB_PATH}")
print(f"Speaker split: train={spk_counts.get('train',0)}  dev={spk_counts.get('dev',0)}  test={spk_counts.get('test',0)}")
print(f"Segment split: train={counts.get('train',0)}  dev={counts.get('dev',0)}  test={counts.get('test',0)}")
print(db)

# -- stress_level distribution per split ------------------------------------
df_plot = df.copy()
df_plot["split"] = pd.array(speaker_ids, dtype="string")
df_plot["split"] = df_plot["split"].map(speaker_split)

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
palette = {"train": "#4878CF", "dev": "#6ACC65", "test": "#D65F5F"}
order   = ["train", "dev", "test"]

fig, ax = plt.subplots(figsize=(9, 4))
for split in order:
    subset = df_plot.loc[df_plot["split"] == split, "stress_level"]
    sns.kdeplot(subset, label=f"{split} (n={len(subset):,})", color=palette[split], ax=ax)
ax.set_xlabel("Mean perceptual stress score")
ax.set_ylabel("Density")
ax.set_title("stress_level distribution per split")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "split_stress_distribution.png", dpi=150)
plt.close(fig)
print(f"\nPlot saved → {RESULTS / 'split_stress_distribution.png'}")
