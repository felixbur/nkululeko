# Nkululeko pre-processing for SONAR dataset

## Description
This directory contains the pre-processing script and configuration for using the SONAR audio deepfake benchmark with the Nkululeko framework.

SONAR ships as one folder of genuine speech (`real_samples`, 2274 LibriTTS test-clean utterances) plus eight folders of synthetic speech from different TTS/voice-generation systems (`AudioGen`, `FlashSpeech`, `NaturalSpeech3`, `OpenAI`, `PromptTTS2`, `VALLE`, `VoiceBox`, `xTTS`), 1674 files total. The audio itself is not stored in this repository (see `--data_dir` below); only the generated CSV metadata is committed.

Class balance (real vs. fake): 2274 vs. 1674 (~1.36:1), so only mildly imbalanced.

`seedtts.csv` (also shipped with SONAR) lists paths into a separate SeedTTS test set that must be downloaded independently (see `~/data/SONAR/README.md`); it is not included in this pre-processing pass.

## Speaker grouping caveat
Splits are speaker-independent where a speaker id is actually recoverable from the filename convention:
- `real_samples` (LibriTTS naming) and `xTTS` (voice cloned from a numbered reference speaker): leading number used as speaker id.
- `OpenAI`: one of a handful of fixed TTS voice names used as speaker id.
- `AudioGen`, `FlashSpeech`, `NaturalSpeech3`, `PromptTTS2`, `VoiceBox`: plain sequential filenames carry no speaker signal.
- `VALLE`: mixes several unrelated naming schemes (LibriSpeech ids, VCTK ids, Fisher ids, emotion labels, bare indices) that can't be resolved reliably.

For the systems with no recoverable speaker id, each file is treated as its own pseudo-speaker, so the speaker-grouped split degrades gracefully to a plain per-file split for those rows only.

## Sample rate caveat
SONAR ships at mixed native sample rates -- mostly 24000 Hz (LibriTTS-derived `real_samples`, `xTTS`, `VALLE`), some 22050 Hz, a few already at 16000 Hz. nkululeko's wav2vec2 extractor (`nkululeko/feat_extract/feats_wav2vec2.py`) asserts exactly 16000 Hz and does not resample -- it silently skips anything else, and with most of SONAR off-rate that blows past the extractor's default 50% failure threshold and aborts. Run `resample_to_16k.py` first to make a 16kHz mono copy, then point `process_database.py` at that copy.

## Commands

```bash
# 1. resample to 16kHz mono (writes to ~/data/SONAR_16k by default)
python data/sonar/resample_to_16k.py

# 2. generate CSVs from the resampled copy
python data/sonar/process_database.py --data_dir ~/data/SONAR_16k

# 3. run the experiment
python -m nkululeko.nkululeko --config data/sonar/exp.ini
```

Pass `--src`/`--dst` to `resample_to_16k.py` and `--data_dir`/`--output_dir` to `process_database.py` if the dataset lives elsewhere.

Reference:
See `~/data/SONAR/README.md` (shipped with the dataset) for provenance and the SeedTTS download link.
