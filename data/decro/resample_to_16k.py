import argparse
import os

import librosa
import soundfile as sf

# DECRO ships at mixed native sample rates per split -- roughly 57%
# already 16000 Hz, the rest 22050 Hz (LJSpeech-derived TTS output),
# 24000 Hz (voice-conversion output), 21900 Hz, and 44100 Hz (spot-
# checked via a 200-file-per-split sample). nkululeko's wav2vec2
# extractor asserts exactly 16000 Hz and does not resample (nkululeko/
# feat_extract/feats_wav2vec2.py), so any non-16kHz file gets skipped
# outright -- with roughly 43% of DECRO off-rate, that skip rate blows
# past the default 50% failure threshold and aborts feature extraction
# (same issue SONAR had -- see data/sonar/resample_to_16k.py). This
# script makes a 16kHz mono copy of all 6 splits so process_database.py
# (pointed at the copy) produces CSVs that actually work with nkululeko.
#
# Every file is re-encoded uniformly through librosa+soundfile, even the
# already-16kHz ones -- guarantees a consistent PCM_16 encoding across
# the whole copy rather than leaving some files in whatever original
# encoding they shipped with.

SPLITS = ["en_train", "en_dev", "en_eval", "ch_train", "ch_dev", "ch_eval"]


def main(src_dir, dst_dir):
    converted, skipped, failed = 0, 0, 0
    for split in SPLITS:
        src_split_dir = os.path.join(src_dir, split)
        if not os.path.isdir(src_split_dir):
            print(f"WARNING: {src_split_dir} not found, skipping")
            continue
        dst_split_dir = os.path.join(dst_dir, split)
        os.makedirs(dst_split_dir, exist_ok=True)
        files = sorted(
            f for f in os.listdir(src_split_dir) if f.lower().endswith(".wav")
        )
        for i, fname in enumerate(files):
            src_path = os.path.join(src_split_dir, fname)
            dst_path = os.path.join(dst_split_dir, fname)
            if os.path.isfile(dst_path):
                skipped += 1
                continue
            try:
                signal, _ = librosa.load(src_path, sr=16000, mono=True)
                sf.write(dst_path, signal, 16000, subtype="PCM_16")
                converted += 1
            except Exception as e:
                print(f"FAILED: {src_path}: {e}")
                failed += 1
            if (i + 1) % 5000 == 0:
                print(f"{split}: {i + 1}/{len(files)}")
        print(f"{split}: done ({len(files)} files)")

    print(
        f"\nConverted {converted} files, skipped {skipped} already-present, "
        f"{failed} failed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample DECRO audio to 16kHz mono (required by nkululeko's wav2vec2 extractor)"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="/home/bagus/data/DECRO/petrichorwq-DECRO-dataset-6fc9884",
        help="Path to the extracted DECRO dataset directory",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="/home/bagus/data/DECRO_16k",
        help="Path to write the 16kHz-resampled copy",
    )
    args = parser.parse_args()

    main(args.src, args.dst)
