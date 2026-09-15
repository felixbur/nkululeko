import argparse
import os

import soundfile as sf
import librosa

# SONAR ships its nine subfolders at mixed native sample rates (mostly
# 24000 Hz for LibriTTS-derived real_samples/xTTS/VALLE, some 22050 Hz,
# a few already at 16000 Hz). nkululeko's wav2vec2 extractor asserts
# exactly 16000 Hz and does not resample (nkululeko/feat_extract/
# feats_wav2vec2.py), so any non-16kHz file gets skipped outright -- with
# most of SONAR off-rate, that skip rate blows past the default 50%
# failure threshold and aborts the whole feature extraction. This script
# makes a 16kHz mono copy of the dataset so process_database.py (pointed
# at the copy) produces CSVs that actually work with nkululeko.

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


def main(src_dir, dst_dir):
    converted, skipped = 0, 0
    for system in [REAL_SYSTEM] + FAKE_SYSTEMS:
        src_sys_dir = os.path.join(src_dir, system)
        if not os.path.isdir(src_sys_dir):
            print(f"WARNING: {src_sys_dir} not found, skipping")
            continue
        dst_sys_dir = os.path.join(dst_dir, system)
        os.makedirs(dst_sys_dir, exist_ok=True)
        for fname in sorted(os.listdir(src_sys_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            src_path = os.path.join(src_sys_dir, fname)
            dst_path = os.path.join(dst_sys_dir, fname)
            if os.path.isfile(dst_path):
                skipped += 1
                continue
            signal, _ = librosa.load(src_path, sr=16000, mono=True)
            sf.write(dst_path, signal, 16000, subtype="PCM_16")
            converted += 1
        print(f"{system}: done")

    print(f"\nConverted {converted} files, skipped {skipped} already-present files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample SONAR audio to 16kHz mono (required by nkululeko's wav2vec2 extractor)"
    )
    parser.add_argument(
        "--src",
        type=str,
        default=os.path.expanduser("~/data/SONAR"),
        help="Path to the original SONAR dataset directory",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=os.path.expanduser("~/data/SONAR_16k"),
        help="Path to write the 16kHz-resampled copy",
    )
    args = parser.parse_args()

    main(args.src, args.dst)
