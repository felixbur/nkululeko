# Nkululeko pre-processing for CVoiceFake-small dataset

## Description
This directory contains the pre-processing script and configuration for using the CVoiceFake-small audio deepfake dataset with the Nkululeko framework.

The dataset is organized by language (`de`, `en`, `fr`, `it`, `zh-CN`); each language folder has a `Bonafide/` subfolder of genuine Common Voice clips and five subfolders of spoofed counterparts from different vocoder/synthesis systems (`griffin_lim_generated`, `vctk_multi_band_melgan.v2_generated`, `vctk_parallel_wavegan.v1_generated`, `vctk_style_melgan.v1_generated`, `world_generated`). All languages are combined into one train/dev/test split. The audio itself is not stored in this repository; only the generated CSV metadata is committed.

Class balance (real vs. fake): 23,544 vs. 114,592 (~1:4.9) -- **heavily imbalanced toward fake**, since each real clip has five synthetic counterparts. Plan on class-weighted loss / focal loss / resampling (see `data/itw/exp_weighted.ini` and `data/itw/exp_focal.ini` for worked examples in this repo) rather than the plain-BCE baseline in `exp.ini` if you want a well-calibrated classifier rather than just an EER benchmark.

## Speaker grouping caveat
No speaker manifest (e.g. Common Voice's `client_id`) ships with this dataset -- filenames only carry the original Common Voice clip id, which is not a speaker id. Every file is therefore treated as its own pseudo-speaker, so the "speaker-grouped" split in `process_database.py` degenerates to a stratified random file-level split. If you need a genuinely speaker-independent split (e.g. for a publication claim), fetch the matching Common Voice `validated.tsv` for each language separately and join on clip id to recover real `client_id` groups.

## Commands

```bash
# audio already present at ~/data/CVoiceFake_small (default --data_dir)
python data/cvoicefake_small/process_database.py
python -m nkululeko.nkululeko --config data/cvoicefake_small/exp.ini
```

Pass `--data_dir /path/to/CVoiceFake_small` if the dataset lives elsewhere, and `--output_dir` to write the CSVs somewhere other than this directory.

Note: the combined train split has ~83k files; wav2vec2 feature extraction over the full set will take substantially longer than the other deepfake datasets in this repo (`data/itw`, `data/sonar`).
