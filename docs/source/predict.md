# Nkululeko Predict Module

`nkululeko.predict` is the unified prediction module of Nkululeko. It replaces
the previous `nkululeko.demo`, `nkululeko.feature_demo` and `nkululeko.testing`
modules and bundles all of their functionality behind a single command-line
interface.

You can use it to predict labels for:

- one or more individual audio files (`--file`)
- every audio file inside a folder (`--folder`)
- the audio paths listed in a CSV (`--list`, original columns are preserved)
- a live microphone recording (`--mic`)

…using one of two prediction sources:

- a **feature extractor** or **autopredict target** such as `age`, `gender`,
  `emotion`, `mos`, `snr` (`--type feats`, the default)
- the **best model from a previously trained experiment**
  (`--type model`, requires `--config`)

## Command-line interface

```text
python -m nkululeko.predict
    [--file AUDIO [AUDIO ...] | --folder FOLDER | --list CSV | --mic]
    [--model MODEL] [--type {feats,model}]
    [--config CONFIG.ini] [--outfile OUTFILE]
```

| Argument | Description |
|---|---|
| `--file AUDIO [AUDIO ...]` | One or more audio files. A single space-separated string also works (e.g. `--file "a.wav b.wav"`). Writes a per-file `<name>_result.txt` next to each input and prints results to stdout. |
| `--folder FOLDER` | Folder to scan recursively for audio (`wav`, `mp3`, `flac`, `ogg`, `m4a`, `au`, `aac`). Writes a single CSV to `--outfile`. |
| `--list CSV` | CSV with audio paths. Existing columns and the audformat index are preserved; prediction columns are appended. Writes a single CSV to `--outfile`. |
| `--mic` | Record `5` seconds from the microphone in a loop and print predictions to stdout. |
| `--model MODEL` | Either an autopredict target name (`age`, `gender`, `emotion`, `mos`, `snr`, `pesq`, `sdr`, `stoi`, `arousal`, `valence`, `dominance`, `speaker`, `text`, `textclassification`, `translation`) **or** a feature-extractor name (`wav2vec2-...`, `opensmile`, `audmodel`, `emotion2vec-...`, `praat`, `clap`, `spkrec`, `trill`, `agender`, `whisper-...`, `ast`, `hubert-...`, `wavlm-...`, `squim`, `mos`, `snr`). When `--type model`, `--model` is ignored — the trained model from the experiment is used. |
| `--type {feats,model}` | `feats` (default): use `--model` as autopredict target or feature extractor. `model`: load the best model from the experiment defined by `--config`. |
| `--config CONFIG.ini` | Optional INI file. Required for `--type model`. With `--type feats` it may supply `FEATS.type` so that `--model` can be omitted. |
| `--outfile OUTFILE` | Output CSV path for `--list` and `--folder`. Default: `./prediction_result.csv`. |

The four input arguments (`--file`, `--folder`, `--list`, `--mic`) are mutually
exclusive.

## Examples

### Predict emotion for a couple of audio files

```bash
python -m nkululeko.predict --file test.mp3 test2.wav --model emotion
```

This writes `test_result.txt` and `test2_result.txt` next to each input and
also prints the predictions to stdout. With `--model emotion`, the
`nkululeko.autopredict.ap_emotion` predictor is used.

### Predict SNR for every audio file in a folder

```bash
python -m nkululeko.predict --folder ./recordings --model snr --outfile snr.csv
```

The output CSV contains the audformat segmented index plus the new
`snr_pred` column.

### Add prediction columns to an existing CSV, keeping all original columns

```bash
python -m nkululeko.predict \
    --list testdata.csv \
    --model mos \
    --outfile testdata_with_mos.csv
```

If `testdata.csv` is a valid audformat CSV (segmented or filewise index), the
index is preserved. Otherwise the first column is interpreted as the audio
path. Any further columns are passed through to the output.

### Use the best model of a trained experiment

```bash
python -m nkululeko.predict \
    --list testdata.csv \
    --config config.ini \
    --type model
```

This loads the experiment specified in `config.ini` (which must have been
trained with `MODEL.save = True`) and runs its best model on each file in the
list. For classification, the output contains one column per class label with
the probability/score and a `predicted` column with the top-1 label. For
regression, a single `predicted` column is written.

### Loop over microphone input using the FEATS section of a config

```bash
python -m nkululeko.predict --mic --config config.ini
```

Press *Enter* to record `5` seconds, *q* + *Enter* to quit.

## Autopredict targets

When `--model NAME` matches one of the autopredict targets below, the matching
`nkululeko.autopredict.*` predictor is used. The added column name follows the
`<target>_pred` convention.

| Target | Predictor module | Added column |
|---|---|---|
| `speaker` | `ap_sid.SIDPredictor` | `speaker_pred` |
| `gender` | `ap_gender.GenderPredictor` (audEERING agender) | `gender_pred` |
| `age` | `ap_age.AgePredictor` (audEERING agender) | `age_pred` |
| `emotion` | `ap_emotion.EmotionPredictor` (emotion2vec) | `emotion_pred` |
| `arousal` | `ap_arousal.ArousalPredictor` (audEERING dim) | `arousal_pred` |
| `valence` | `ap_valence.ValencePredictor` (audEERING dim) | `valence_pred` |
| `dominance` | `ap_dominance.DominancePredictor` (audEERING dim) | `dominance_pred` |
| `mos` | `ap_mos.MOSPredictor` | `mos_pred` |
| `pesq` | `ap_pesq.PESQPredictor` (SQUIM) | `pesq_pred` |
| `sdr` | `ap_sdr.SDRPredictor` (SQUIM) | `sdr_pred` |
| `stoi` | `ap_stoi.STOIPredictor` (SQUIM) | `stoi_pred` |
| `snr` | `ap_snr.SNRPredictor` | `snr_pred` |
| `text` | `ap_text.TextPredictor` (whisper transcription) | `text` |
| `textclassification` | `ap_textclassifier.TextClassificationPredictor` | `classification_winner` + one column per candidate label |
| `translation` | `ap_translate.TextTranslator` | column named after `PREDICT.target_language` (default: `en`) |

## Feature extractors

If `--model` does **not** match an autopredict target, it is interpreted as a
feature-extractor name. The output columns are `feat_0`, `feat_1`, …. Examples:

```bash
python -m nkululeko.predict --file test.wav --model praat
python -m nkululeko.predict --folder ./voices --model wav2vec2-large-robust-ft-swbd-300h --outfile feats.csv
python -m nkululeko.predict --list audio.csv --model audmodel --outfile feats.csv --config has_audmodel_id.ini
```

Recognized prefixes / names: `wav2vec2*`, `hubert*`, `wavlm*`, `whisper*`,
`ast*`, `emotion2vec*`, `opensmile`/`gemaps`/`compare`, `clap*`, `spkrec*` /
`xvect*` / `ecapa*`, `trill*`, `praat*`, `audmodel*`, `agender*`, `squim*` /
`pesq*` / `sdr*`, `mos*`, `snr*`.

> **Note on overlapping names.** `mos` and `snr` are both autopredict targets
> *and* feature extractors. They resolve to the autopredict path. If you need
> the raw feature extractor for these, use the lower-level extractor classes
> directly.

## Output formats

| Mode | Where the result is written |
|---|---|
| `--file` | `<name>_result.txt` per input file (one `key: value` per line), plus stdout. |
| `--folder` | Single CSV at `--outfile` with the audformat segmented index of the discovered files and the prediction columns. |
| `--list` | Single CSV at `--outfile` with the original columns of the input CSV plus the prediction columns. The audformat index is preserved when the input is a valid audformat CSV. |
| `--mic` | stdout only. |

## See also

- [demo.md](demo.md) — tutorial for using a previously trained model (`--type model`).
- [emotion_prediction.md](emotion_prediction.md) — predicting emotions on unlabeled audio.
- [predict_speaker.md](predict_speaker.md) — predicting speaker identity.
- [text_processing.md](text_processing.md) — transcription, translation and text classification.
