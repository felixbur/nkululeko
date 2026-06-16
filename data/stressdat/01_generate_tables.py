# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "audformat>=1.0",
#     "audiofile>=1.0",
#     "matplotlib>=3.9",
#     "pandas>=2.0",
#     "seaborn>=0.13",
#     "silero-vad>=5.0",
#     "torch>=2.0",
# ]
# ///
"""
Generate a per-utterance metadata CSV for the StressDat corpus.

Source of truth: StressDat-anotacia-29spk_v1_normalized.txt (recommended per README).
Only utterances present in that file are included.

Columns:
  induced_stress – intended stress level from the filename suffix (0=neutral, 1=medium, 2=high)
  gender         – speaker gender (female / male) from README speaker table
  stress_level   – mean perceptual stress score across annotators
  duration       – audio duration in seconds

Also produces stressdat_segments.csv: same columns but with a VAD-based segmented
index (file, start, end in seconds), where duration is the speech segment length.

Index: relative file path from the repo root (POSIX style, from StressDat_all.scp)
"""

import re
from pathlib import Path

import audformat
import audiofile
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from audformat import segmented_index

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)
SAMPLING_RATE = 16000


def get_segmentation_simple(file):
    get_speech_timestamps, _, read_audio, _, _ = vad_utils
    wav = read_audio(str(file), sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLING_RATE)
    files, starts, ends = [], [], []
    for entry in speech_timestamps:
        files.append(str(file))
        starts.append(float(entry["start"] / SAMPLING_RATE))
        ends.append(float(entry["end"] / SAMPLING_RATE))
    return segmented_index(files, starts, ends)

REPO = Path(__file__).parent / "repo"

# --------------------------------------------------------------------------- #
# 1.  Speaker gender lookup  (from README / metadata table)
# --------------------------------------------------------------------------- #
GENDER = {
    "s01": "female", "s02": "male",   "s03": "male",   "s04": "female", "s05": "male",
    "s06": "male",   "s07": "male",   "s08": "female", "s09": "male",   "s10": "female",
    "s11": "female", "s12": "female", "s13": "female", "s14": "female", "s15": "male",
    "s16": "male",   "s17": "male",   "s18": "male",   "s19": "female", "s20": "male",
    "s21": "female", "s22": "male",   "s23": "female", "s24": "female", "s25": "male",
    "s26": "male",   "s27": "female", "s28": "female", "s29": "female", "s30": "female",
}

INDUCED_STRESS_MAP = {
    "a": 0,  # neutral
    "b": 1,  # medium stress
    "c": 2,  # high stress
}

LEVEL_RE = re.compile(r"^(s\d+[a-z]+)_\d+([abc])_")

# --------------------------------------------------------------------------- #
# 2.  Load normalized annotations and compute per-file mean score
# --------------------------------------------------------------------------- #
annot = pd.read_csv(REPO / "StressDat-anotacia-29spk_v1_normalized.txt", sep="\t")
mean_scores = annot.groupby("FILE")["SCORE"].mean()

# --------------------------------------------------------------------------- #
# 3.  Build utterance-id → POSIX path lookup from the SCP file
# --------------------------------------------------------------------------- #
scp = {}
with open(REPO / "StressDat_all.scp") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        utt_id, win_path = line.split("\t")
        scp[utt_id] = win_path.replace("\\", "/")

# --------------------------------------------------------------------------- #
# 4.  Build rows — one per utterance in the normalized dataset
# --------------------------------------------------------------------------- #
rows = {}
for utt_id, score in mean_scores.items():
    m = LEVEL_RE.match(utt_id)
    if not m:
        continue
    full_speaker = m.group(1)       # e.g. "s06h"
    level_char   = m.group(2)       # e.g. "b"
    speaker_key  = full_speaker[:3] # e.g. "s06"

    rel_path = scp.get(utt_id)
    if rel_path is None:
        continue

    rows["audio/" + rel_path] = {
        "induced_stress": INDUCED_STRESS_MAP[level_char],
        "gender":         GENDER.get(speaker_key),
        "stress_level":   score,
        "duration":       audiofile.duration(REPO / rel_path),
    }

df = pd.DataFrame.from_dict(rows, orient="index")
df.index.name = "file_path"

# --------------------------------------------------------------------------- #
# 5.  Save
# --------------------------------------------------------------------------- #
out_path = Path(__file__).parent / "stressdat_files.csv"
df.to_csv(out_path)
print(f"Saved {len(df):,} rows → {out_path}")
print(df.head())
print(f"\nInduced stress counts:\n{df['induced_stress'].value_counts().sort_index()}")
print(f"\nGender counts:\n{df['gender'].value_counts()}")

# --------------------------------------------------------------------------- #
# 5b. VAD speech-segment table
# --------------------------------------------------------------------------- #
print("\nRunning silero VAD …")
seg_rows = []
for rel_path in df.index:
    # df.index uses "audio/<rel>" keys; strip prefix for physical path
    bare = rel_path[len("audio/"):]
    full_path = REPO / bare
    seg_idx = get_segmentation_simple(full_path)
    if len(seg_idx) == 0:
        # no speech detected — keep the full file as one segment
        dur = audiofile.duration(full_path)
        seg_rows.append({"file": rel_path, "start": 0.0, "end": dur})
        continue
    starts = seg_idx.get_level_values("start")
    ends   = seg_idx.get_level_values("end")
    for s, e in zip(starts, ends):
        seg_rows.append({
            "file":  rel_path,
            "start": s.total_seconds() if hasattr(s, "total_seconds") else float(s),
            "end":   e.total_seconds() if hasattr(e, "total_seconds") else float(e),
        })

seg_df = pd.DataFrame(seg_rows)
seg_df["induced_stress"] = seg_df["file"].map(df["induced_stress"])
seg_df["gender"]         = seg_df["file"].map(df["gender"])
seg_df["stress_level"]   = seg_df["file"].map(df["stress_level"])
seg_df["duration"]       = seg_df["end"] - seg_df["start"]

seg_path = Path(__file__).parent / "stressdat_segments.csv"
seg_df.to_csv(seg_path, index=False)
print(f"Saved {len(seg_df):,} segments → {seg_path}")

# --------------------------------------------------------------------------- #
# 6.  Analysis plots
# --------------------------------------------------------------------------- #
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

LEVEL_LABELS = {0: "neutral", 1: "medium", 2: "high"}
PALETTE = sns.color_palette("muted")

sns.set_theme(style="whitegrid", font_scale=1.1)

# -- 6a. Histogram of perceptual stress score --------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(df["stress_level"], bins=30, kde=True, color=PALETTE[0], ax=ax)
ax.set_xlabel("Mean perceptual stress score")
ax.set_ylabel("Utterance count")
ax.set_title("Distribution of perceptual stress scores (annotator mean)")
fig.tight_layout()
fig.savefig(RESULTS / "hist_stress_level.png", dpi=150)
plt.close(fig)

# -- 6b. Bar chart of induced stress level counts ----------------------------
level_counts = (
    df["induced_stress"]
    .value_counts()
    .sort_index()
    .rename(index=LEVEL_LABELS)
)
level_df = level_counts.reset_index()
level_df.columns = ["level", "count"]
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=level_df, x="level", y="count", hue="level", palette=PALETTE[:len(level_df)], legend=False, ax=ax)
ax.set_xlabel("Induced stress level")
ax.set_ylabel("Utterance count")
ax.set_title("Utterance count per induced stress level")
for bar, val in zip(ax.patches, level_df["count"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f"{val:,}",
        ha="center", va="bottom", fontsize=10,
    )
fig.tight_layout()
fig.savefig(RESULTS / "bar_induced_stress.png", dpi=150)
plt.close(fig)

# -- 6c. Gender distribution -------------------------------------------------
gender_counts = df["gender"].value_counts()
gender_df = gender_counts.reset_index()
gender_df.columns = ["gender", "count"]
fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(data=gender_df, x="gender", y="count", hue="gender", palette=PALETTE[2:2+len(gender_df)], legend=False, ax=ax)
ax.set_xlabel("Gender")
ax.set_ylabel("Utterance count")
ax.set_title("Utterance count by gender")
for bar, val in zip(ax.patches, gender_df["count"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f"{val:,}",
        ha="center", va="bottom", fontsize=10,
    )
fig.tight_layout()
fig.savefig(RESULTS / "bar_gender.png", dpi=150)
plt.close(fig)

# -- 6d. Stress score vs induced level (box plot + strip) --------------------
df_plot = df.copy()
df_plot["induced_label"] = df_plot["induced_stress"].map(LEVEL_LABELS)
order = [LEVEL_LABELS[k] for k in sorted(df["induced_stress"].unique())]

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df_plot, x="induced_label", y="stress_level",
    order=order, hue="induced_label", palette=PALETTE[:len(order)], legend=False,
    width=0.5, linewidth=1.2, ax=ax,
)
sns.stripplot(
    data=df_plot, x="induced_label", y="stress_level",
    order=order, color="black", alpha=0.08, size=2, jitter=True, ax=ax,
)
r = df["stress_level"].corr(df["induced_stress"])
ax.set_xlabel("Induced stress level")
ax.set_ylabel("Mean perceptual stress score")
ax.set_title(f"Perceptual score vs induced level  (Pearson r = {r:.3f})")
fig.tight_layout()
fig.savefig(RESULTS / "corr_stress_vs_induced.png", dpi=150)
plt.close(fig)

# -- 6e. Utterances and speakers per speaker, coloured by gender -------------
speaker_col = pd.Series(
    [Path(p).parts[1] for p in df.index], index=df.index, name="speaker"
)
df_spk = df.copy()
df_spk["speaker"] = speaker_col
speaker_gender = df_spk.groupby("speaker")["gender"].first()
speaker_counts = df_spk.groupby("speaker").size().sort_index()

spk_df = pd.DataFrame({
    "speaker": speaker_counts.index,
    "count":   speaker_counts.values,
    "gender":  speaker_gender[speaker_counts.index].values,
})

gender_colors = {"female": PALETTE[2], "male": PALETTE[0]}
bar_colors = [gender_colors[g] for g in spk_df["gender"]]

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(spk_df["speaker"], spk_df["count"], color=bar_colors, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Speaker")
ax.set_ylabel("Utterance count")
n_spk = spk_df["speaker"].nunique()
ax.set_title(f"Utterances per speaker  ({n_spk} speakers, {len(df):,} total samples)")
ax.tick_params(axis="x", rotation=45)
from matplotlib.patches import Patch
ax.legend(
    handles=[Patch(color=gender_colors["female"], label="female"),
             Patch(color=gender_colors["male"],   label="male")],
    title="Gender",
)
fig.tight_layout()
fig.savefig(RESULTS / "bar_speakers.png", dpi=150)
plt.close(fig)

# -- 6f. Duration vs induced stress level ------------------------------------
r_dur = df["duration"].corr(df["induced_stress"])

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(
    data=df_plot, x="induced_label", y="duration",
    order=order, hue="induced_label", palette=PALETTE[:len(order)], legend=False,
    width=0.5, linewidth=1.2, ax=ax,
)
sns.stripplot(
    data=df_plot, x="induced_label", y="duration",
    order=order, color="black", alpha=0.08, size=2, jitter=True, ax=ax,
)
ax.set_xlabel("Induced stress level")
ax.set_ylabel("Duration (s)")
ax.set_title(f"Utterance duration vs induced level  (Pearson r = {r_dur:.3f})")
fig.tight_layout()
fig.savefig(RESULTS / "corr_duration_vs_induced.png", dpi=150)
plt.close(fig)

# -- 6g. Distribution of VAD segment durations -------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(seg_df["duration"], bins=40, kde=True, color=PALETTE[1], ax=ax)
ax.set_xlabel("Segment duration (s)")
ax.set_ylabel("Segment count")
ax.set_title(
    f"Distribution of VAD speech segment durations  "
    f"(n={len(seg_df):,}, median={seg_df['duration'].median():.2f}s)"
)
fig.tight_layout()
fig.savefig(RESULTS / "hist_segment_duration.png", dpi=150)
plt.close(fig)

print(f"\nPlots saved to {RESULTS}/")
print(f"  hist_stress_level.png")
print(f"  bar_induced_stress.png")
print(f"  bar_gender.png")
print(f"  corr_stress_vs_induced.png")
print(f"  bar_speakers.png")
print(f"  corr_duration_vs_induced.png")
print(f"  hist_segment_duration.png")
print(f"\nPearson r (stress_level ~ induced_stress): {r:.4f}")
print(f"Pearson r (duration     ~ induced_stress): {r_dur:.4f}")
