# DECRO - DEepfake CROss-lingual evaluation dataset

## Overview

**DECRO** is built specifically to evaluate the influence of language differences on audio deepfake detection. It has English and Chinese subsets of roughly matched size, and — most importantly — the spoofed speech in both subsets is generated with the **same types of synthesis algorithms** (HiFiGAN, Multiband-MelGAN, PWG, Tacotron, FastSpeech2, VITS, Starganv2-vc, plus commercial Baidu/Xunfei TTS). That design isolates the language axis from the spoofing-method axis, unlike a corpus where different languages happen to use different vocoders.

## Dataset Information

- **Languages**: English, Chinese
- **License**: see `LICENSE` in the extracted archive
- **Access**: public (Zenodo)
- **Target**: cross-lingual deepfake detection
- **Paper**: Ba, Wen, Cheng, Wang, Lin, Lu, Liu. "Transferring Audio Deepfake Detection Capability across Languages." WWW 2023.

## Composition

|  | English Train | English Dev | English Eval | Chinese Train | Chinese Dev | Chinese Eval |
|---|---|---|---|---|---|---|
| Bona-fide | 5129 | 3049 | 4306 | 9000 | 6109 | 6109 |
| Spoofed | 17412 | 10503 | 14884 | 17850 | 12015 | 12015 |
| Total | 22541 | 13552 | 19190 | 26850 | 18124 | 18124 |

Chinese bona-fide audio is drawn from six open-source recording corpora (Aidatatang_200zh, AISHELL-1/2/3, freeST, MagicData). English bona-fide audio is drawn from ASVspoof2019 LA, redivided to fit this dataset's split sizes.

## Download

- **URL**: https://zenodo.org/record/7603208
- **File**: `petrichorwq/DECRO-dataset-v1.2.zip` (~15.5 GB zipped, ~19.4 GB extracted, 118,396 audio files)
- **GitHub**: https://github.com/petrichorwq/DECRO-dataset

## Citation

```bibtex
@inproceedings{ba2023transferring,
  title={Transferring Audio Deepfake Detection Capability across Languages},
  author={Ba, Zhongjie and Wen, Qing and Cheng, Peng and Wang, Yuwei and Lin, Feng and Lu, Li and Liu, Zhenguang},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={2033--2044},
  year={2023}
}
```

## Data Format

Single-channel WAV files, split into 6 directories: `{en,ch}_{train,dev,eval}/`. Each split has a matching ASVspoof-style protocol file `{en,ch}_{train,dev,eval}.txt`:

```
SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
```

`AUDIO_FILE_NAME` has no extension; `KEY` is `bonafide` or `spoof`.

**Sample rate caveat**: DECRO ships at mixed native rates -- roughly 57% already 16000 Hz, the rest 22050 Hz (LJSpeech-derived TTS output), 24000 Hz (voice-conversion output), 21900 Hz, or 44100 Hz. nkululeko's wav2vec2 extractor requires exactly 16000 Hz and skips (doesn't resample) anything else, which blows past its default 50% failure threshold on this dataset. Run `resample_to_16k.py` first (same fix SONAR needed -- see `data/sonar/resample_to_16k.py`) to get a usable 16kHz mono copy.

## Usage with Nkululeko

1. Download and extract the zip to `/home/bagus/data/DECRO/` (or update `--src`/`--data_dir` below), producing `petrichorwq-DECRO-dataset-<hash>/`.
2. Run `python resample_to_16k.py` to write a 16kHz mono copy to `/home/bagus/data/DECRO_16k/`.
3. Run `python process_database.py` from this directory (defaults to reading the resampled copy) to generate:
   - `decro_en_train.csv`, `decro_en_dev.csv`, `decro_en_eval.csv`
   - `decro_ch_train.csv`, `decro_ch_dev.csv`, `decro_ch_eval.csv`
   - `decro_en.csv`, `decro_ch.csv` (each language's 3 splits combined)
   - `decro.csv` (everything combined)
4. Columns: `file, label, language, system, speaker`. `label` is `real`/`fake` (mapped from `bonafide`/`spoof`).

For the paper's actual cross-lingual protocol: train on one language's train+dev, evaluate on the *other* language's eval set (e.g. train on `decro_en_train.csv`+`decro_en_dev.csv`, test on `decro_ch_eval.csv`, and vice versa).
