#!/usr/bin/env python3
"""Export CLSE (Cognitive Load with Speech and EGG) to nkululeko CSV file lists.

Reads from the raw ComParE 2014 distribution directory
(the "extracted_and_merged" folder from the NAS at
database/DBS-Public/ComParE2014-cognitive-load).

Requires: silero-vad  (pip install silero-vad)

Labels
------
  objective_cognitive_load  1=low, 2=medium, 3=high  (ComParE 2014 challenge target)
  subjective_cognitive_load 1-9 Likert scale (self-reported mental effort)

Outputs (in --out_dir, default ".")
-------
  clse_train.csv, clse_dev.csv, clse_test.csv
  clse_all.csv  (all three splits combined, with a "split" column)
"""

import argparse
import os

import pandas as pd


SR = 16000  # all CLSE files are 16 kHz


def run_silero_vad(files, min_speech_ms=760, max_speech_s=6.0):
    """Return a DataFrame with (file, start, end) rows in seconds."""
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

    model = load_silero_vad()
    rows = []
    for fpath in files:
        wav = read_audio(fpath, sampling_rate=SR)
        timestamps = get_speech_timestamps(
            wav, model,
            sampling_rate=SR,
            min_speech_duration_ms=min_speech_ms,
            max_speech_duration_s=max_speech_s,
            return_seconds=True,
        )
        for ts in timestamps:
            rows.append({"file": fpath, "start": ts["start"], "end": ts["end"]})
    return pd.DataFrame(rows, columns=["file", "start", "end"])


def whole_file_segments(files):
    """Return a DataFrame covering each file as a single segment."""
    import soundfile as sf

    rows = []
    for fpath in files:
        info = sf.info(fpath)
        rows.append({"file": fpath, "start": 0.0, "end": info.duration})
    return pd.DataFrame(rows, columns=["file", "start", "end"])


def main():
    parser = argparse.ArgumentParser(
        description="Export CLSE to nkululeko CSV file lists."
    )
    parser.add_argument(
        "data_dir",
        help="Path to the raw distribution 'extracted_and_merged' directory.",
    )
    parser.add_argument(
        "--out_dir", default=".", help="Output directory (default: current dir)"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load train/dev labels ---
    df = pd.read_csv(os.path.join(data_dir, "labels.txt"), delimiter=";")
    df[["partition", "speaker", "speech_task"]] = df["Filename"].str.extract(
        r"^([a-zA-Z]+)_(subj\d+)_([a-zA-Z]+)\d+"
    )
    df.rename(
        columns={
            "Filename": "file",
            "objective_label(1-3)": "objective_cognitive_load",
            "subjective_label(1-9)": "subjective_cognitive_load",
        },
        inplace=True,
    )
    df["partition"] = df["partition"].map({"training": "train", "validation": "dev"})

    wav_dir = os.path.join(data_dir, "CLSE_subchallenge_wave")
    df["file"] = df["file"].apply(lambda f: os.path.join(wav_dir, f + ".wav"))

    # --- Load test labels (ARFF format) ---
    test_labels_path = os.path.join(data_dir, "cognitive_load.test")
    with open(test_labels_path) as f:
        lines = f.readlines()
    data_start = next(i for i, ln in enumerate(lines) if ln.strip() == "@data") + 1
    df_test_lbl = pd.read_csv(
        test_labels_path,
        skiprows=data_start,
        header=None,
        names=["file", "objective_cognitive_load", "subjective_cognitive_load"],
    )
    df_test_lbl["file"] = df_test_lbl["file"].str.strip("'")
    df_test_lbl["objective_cognitive_load"] = (
        df_test_lbl["objective_cognitive_load"].str.replace("L", "").astype(int)
    )

    testblind_dir = os.path.join(data_dir, "CLSE_subchallenge_wave_testblind")
    df_test = pd.DataFrame(
        {"file": sorted(f for f in os.listdir(testblind_dir)
                        if os.path.isfile(os.path.join(testblind_dir, f)))}
    )
    df_test = df_test.merge(df_test_lbl, on="file", how="left")
    df_test["file"] = df_test["file"].apply(lambda f: os.path.join(testblind_dir, f))
    df_test["partition"] = "test"
    df_test["speech_task"] = df_test["file"].str.extract(r"_([a-zA-Z]+)\d+\.wav$")

    # --- UBM files (story-reading, one per speaker) ---
    ubm_rows = []
    for split, ubm_dir in [
        ("train", os.path.join(data_dir, "UBM_training")),
        ("dev",   os.path.join(data_dir, "UBM_validation")),
    ]:
        for fname in sorted(os.listdir(ubm_dir)):
            fpath = os.path.join(ubm_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ubm_rows.append({
                "file": fpath,
                "partition": split,
                "speaker": fname.replace("UBM_", "").replace(".wav", ""),
                "speech_task": "storyreading",
                "objective_cognitive_load": 1,
                "subjective_cognitive_load": pd.NA,
            })
    df_ubm = pd.DataFrame(ubm_rows)

    # Combine all
    keep = ["file", "partition", "speaker", "speech_task",
            "objective_cognitive_load", "subjective_cognitive_load"]
    df_all = pd.concat(
        [df[keep], df_ubm[keep], df_test[keep]], ignore_index=True
    )
    df_all["objective_cognitive_load"] = df_all["objective_cognitive_load"].astype(
        pd.Int64Dtype()
    )
    df_all["subjective_cognitive_load"] = df_all["subjective_cognitive_load"].astype(
        pd.Int64Dtype()
    )

    # --- VAD strategy per speech task ---
    # default: 760ms min segment, 6s max (sentence-length utterances + story reading)
    # short:   250ms min segment, 6s max (Stroop tasks with short bursts)
    # no_vad:  whole file as one segment (readingspanLetter — continuous reading)
    dct_vad = {
        "default": ["readingspanSentence", "storyreading"],
        "short":   ["stroopdualtask", "strooptimepressure"],
        "no_vad":  ["readingspanLetter"],
    }
    task_to_strategy = {t: k for k, ts in dct_vad.items() for t in ts}
    df_all["strategy"] = df_all["speech_task"].map(task_to_strategy)

    out_splits = []
    for split in ("train", "dev", "test"):
        df_split = df_all[df_all["partition"] == split].copy()
        file_meta = df_split.set_index("file")
        all_segs = []

        groups = {
            "default": df_split[df_split["strategy"] == "default"]["file"].tolist(),
            "short":   df_split[df_split["strategy"] == "short"]["file"].tolist(),
            "no_vad":  df_split[df_split["strategy"] == "no_vad"]["file"].tolist(),
        }

        for strategy, files in groups.items():
            if not files:
                continue
            print(f"  {split}/{strategy}: {len(files)} files")
            if strategy == "no_vad":
                seg_df = whole_file_segments(files)
            elif strategy == "short":
                seg_df = run_silero_vad(files, min_speech_ms=250, max_speech_s=6.0)
            else:
                seg_df = run_silero_vad(files, min_speech_ms=760, max_speech_s=6.0)

            for col in ["speaker", "speech_task",
                        "objective_cognitive_load", "subjective_cognitive_load"]:
                if col in file_meta.columns:
                    seg_df[col] = seg_df["file"].map(file_meta[col].to_dict())
            all_segs.append(seg_df)

        result = pd.concat(all_segs, ignore_index=True)
        result["split"] = split

        out_path = os.path.join(args.out_dir, f"clse_{split}.csv")
        result.to_csv(out_path, index=False)
        print(f"{split}: {len(result)} segments -> {out_path}")
        out_splits.append(result)

    all_path = os.path.join(args.out_dir, "clse_all.csv")
    pd.concat(out_splits, ignore_index=True).to_csv(all_path, index=False)
    print(f"all:   {sum(len(d) for d in out_splits)} segments -> {all_path}")


if __name__ == "__main__":
    main()
