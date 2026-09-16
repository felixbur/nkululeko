import argparse
import os
import re

import pandas as pd

# SONAR audio deepfake benchmark: one "real" folder plus several TTS/voice
# generation systems, each contributing "fake" samples.
REAL_SYSTEM = "real_samples"
FAKE_SYSTEMS = [
    "AudioGen",
    "FlashSpeech",
    "NaturalSpeech3",
    "OpenAI",
    "PromptTTS2",
    "VALLE",
    "VoiceBox",
    "xTTS",
]

LIBRITTS_RE = re.compile(r"^(\d+)_")
OPENAI_RE = re.compile(r"^([a-z]+)_\d+\.wav$")
XTTS_RE = re.compile(r"^(\d+)_")


def _speaker_for_file(system, fname):
    """Best-effort speaker id extraction per generator system.

    real_samples (LibriTTS) and xTTS (voice cloned from a numbered reference
    speaker) encode a speaker id as a leading number; OpenAI uses one of a
    handful of fixed voice names. AudioGen/FlashSpeech/NaturalSpeech3/
    PromptTTS2/VoiceBox use plain sequential indices with no speaker signal,
    and VALLE mixes several unrelated naming schemes (librispeech ids, vctk
    ids, fisher ids, emotion labels, bare indices) that can't be resolved
    reliably. For all of these, fall back to a unique per-file pseudo-speaker
    so the speaker-grouped split below degrades gracefully to a plain
    file-level split instead of risking a wrong grouping.
    """
    if system == REAL_SYSTEM:
        m = LIBRITTS_RE.match(fname)
        if m:
            return f"libritts_{m.group(1)}"
    elif system == "OpenAI":
        m = OPENAI_RE.match(fname)
        if m:
            return f"openai_{m.group(1)}"
    elif system == "xTTS":
        m = XTTS_RE.match(fname)
        if m:
            return f"xtts_{m.group(1)}"
    return f"{system}_{fname}"


def main(data_dir, output_dir):
    """Process the SONAR audio deepfake dataset into nkululeko train/dev/test CSVs.

    Expects the folder layout shipped by SONAR: one subfolder per generation
    system (real_samples for genuine audio, the rest synthetic), each holding
    .wav files directly. The audio stays where it is (data_dir); only the
    generated CSVs (file/speaker/label/system metadata) are written into the
    nkululeko repo at output_dir, so the multi-GB audio itself never needs to
    be copied into or committed to the repo.
    """
    rows = []
    for system in [REAL_SYSTEM] + FAKE_SYSTEMS:
        sys_dir = os.path.join(data_dir, system)
        if not os.path.isdir(sys_dir):
            print(f"WARNING: {sys_dir} not found, skipping")
            continue
        label = "real" if system == REAL_SYSTEM else "fake"
        for fname in sorted(os.listdir(sys_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            rows.append(
                {
                    "file": os.path.join(sys_dir, fname),
                    "speaker": _speaker_for_file(system, fname),
                    "label": label,
                    "system": system,
                }
            )

    df = pd.DataFrame(rows)

    # Speaker-independent split (60% train, 20% dev, 20% test).
    # Most fake-system "speakers" above are unique per file, so those files
    # split like a plain stratified-by-count random split; the recoverable
    # speaker groups (real_samples, OpenAI, xTTS) stay intact across splits.
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

    train_file = os.path.join(output_dir, "sonar_train.csv")
    dev_file = os.path.join(output_dir, "sonar_dev.csv")
    test_file = os.path.join(output_dir, "sonar_test.csv")
    all_file = os.path.join(output_dir, "sonar.csv")

    df_train.to_csv(train_file, index=False)
    df_dev.to_csv(dev_file, index=False)
    df_test.to_csv(test_file, index=False)
    df_shuffled.to_csv(all_file, index=False)

    print(
        f"Created {train_file} with {len(df_train)} samples ({len(train_speakers)} speaker groups)"
    )
    print(
        f"Created {dev_file} with {len(df_dev)} samples ({len(dev_speakers)} speaker groups)"
    )
    print(
        f"Created {test_file} with {len(df_test)} samples ({len(test_speakers)} speaker groups)"
    )
    print(f"Created {all_file} with {len(df_shuffled)} samples")

    print("\nLabel distribution in complete dataset:")
    print(df["label"].value_counts())
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
        description="Process SONAR audio deepfake dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.expanduser("~/data/SONAR"),
        help="Path to the extracted SONAR dataset directory (audio stays here)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to write the generated CSVs into (default: this script's dir)",
    )
    args = parser.parse_args()

    main(args.data_dir, args.output_dir)
