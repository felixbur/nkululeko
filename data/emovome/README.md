# Nkululeko pre-processing for the EMOVOME dataset

EMOVOME (Emotional Voice Messages) is a Spanish speech dataset for emotion
recognition **in the wild**: 999 spontaneous voice messages from 100
speakers (gender-balanced, ages 18-55), collected from real conversations
on a messaging app -- not acted or elicited in a lab. Each message was
rated for valence and arousal (3 classes each: negative/neutral/positive,
low/neutral/high) by three non-expert annotators and one expert, whose
ratings were combined into a single label per dimension; the expert also
assigned one of 7 discrete emotion categories. The database also provides
manual and ASR (Amazon Transcribe) transcriptions of every message,
sociodemographic data per speaker, and an official speaker-independent
train/dev/test partition (80/80/20 speakers; no speaker appears in more
than one split).

Reference:
> L. Gómez-Zaragozá, R. del Amor, M. J. Castro-Bleda, V. Naranjo, M.
> Alcañiz Raya, J. Marín-Morales, "EMOVOME Database: Advancing Emotion
> Recognition in Speech Beyond Staged Scenarios", 2024.
> https://doi.org/10.48550/arXiv.2403.02167

Dataset homepage (Zenodo): https://doi.org/10.5281/zenodo.10694370

## Getting the data

The dataset is split into two parts:

1. **Metadata (open access, no agreement needed)**: `emovome.zip` from
   the [Zenodo record](https://doi.org/10.5281/zenodo.10694370) (1.3 GB).
   It contains `labels.csv`, `participants_ids.csv`,
   `transcriptions.csv`, `Baseline_emotion_recognition/partitions.csv`,
   `Questionnaires/sociodemographic_english.csv` -- everything
   `process_database.py` reads -- plus the eGeMAPS features, baseline
   models and the agreement form, but **no audio**.
2. **Audio (restricted)**: the 999 voice messages themselves are not in
   that zip. Sign `EMOVOME_agreement.pdf` (included in the zip) and send
   it to the corresponding author (jamarmo@htech.upv.es); they deliver
   the `Audios` folder, whose `Emotions/` subfolder has one file per
   message, named `<file_id>.ogg` (Ogg Vorbis, 16-bit, 44.1 or 48 kHz).

Download and unzip it into this directory (`data/emovome/`), so that
`labels.csv`, `Audios/`, `Baseline_emotion_recognition/` etc. end up
sitting right here, next to `process_database.py`:

```bash
cd data/emovome
wget "https://zenodo.org/records/10694370/files/emovome.zip?download=1" -O emovome.zip
unzip emovome.zip
rm emovome.zip
```

Once you have the audio too (step 2 above), drop its `Emotions/`
subfolder into `Audios/` here, so you end up with `Audios/Emotions/`
containing the 999 `<file_id>.ogg` files, next to the already-present
(empty) `Audios/Lecture/`.

Then, still from this directory:

```bash
python process_database.py
python -m nkululeko.nkululeko exp.ini
```

`process_database.py` takes no required options: it reads the metadata
from the current directory and, if `Audios/Emotions/` has files in it,
the audio from there too. Run it before the audio arrives to still get
the CSVs with every label/metadata column filled in but an empty `file`
column -- useful to inspect the data (e.g. with `nkululeko.explore`),
but not runnable for training or feature extraction until the audio is
in place. Pass `--zenodo_dir`/`--audio_dir`/`--out_dir` to override any
of this if your own layout differs.

`exp.ini`'s `os` (openSMILE eGeMAPS) features read the audio at its
native 44.1/48 kHz directly -- no separate resampling step needed. If
you switch `FEATS.type` to a torch-based extractor (e.g. `wav2vec2`),
resample to 16 kHz first: `python -m nkululeko.resample --config exp.ini`.
Note that resampling only covers the `train`/`test` splits, not a
separate `dev` one (a current limitation of `nkululeko.resample`), so
resample `emovome_dev.csv`'s audio yourself if you need it (e.g. by
swapping `databases` down to `['emovome_dev']` and `split_strategy =
train` for one resample run).

## Output columns

`emovome_train.csv` / `emovome_dev.csv` / `emovome_test.csv` (the official
partition) and `emovome_all.csv` (all three, with a `split` column):

| column | values | source |
|---|---|---|
| `file` | absolute path to the voice message | `Audios/Emotions/<file_id>.*` |
| `speaker` | alphanumeric participant ID | `participants_ids.csv` |
| `gender` | `female` / `male` | `Questionnaires/sociodemographic_english.csv` |
| `age_range` | e.g. `18-25` (bucketed, not a single age) | `Questionnaires/sociodemographic_english.csv` |
| `valence` | `negative` / `neutral` / `positive` | `labels.csv: valence_C` (combined non-expert + expert) |
| `arousal` | `low` / `neutral` / `high` | `labels.csv: arousal_C` |
| `category` | `happiness`/`neutral`/`anger`/`surprise`/`sadness`/`fear`/`disgust` | `labels.csv: category_E` (expert only) |
| `transcription` | manually corrected transcript | `transcriptions.csv: Manual` |
| `transcription_asr` | raw Amazon Transcribe output | `transcriptions.csv: AWS` |

`exp.ini` targets `valence` (the dataset's main published benchmark); set
`target = arousal` or `target = category` for the other tasks -- update
`labels` accordingly (`category` is imbalanced: 342 happiness / 225
neutral / 199 anger / 118 surprise / 72 sadness / 35 fear / 8 disgust
messages across all 999, so the original paper's category baseline keeps
only happiness/anger/neutral/surprise; adding
`emovome_train.filter = {'category': ['happiness', 'anger', 'neutral', 'surprise']}`
(and the same for `emovome_dev`/`emovome_test`) reproduces that.)

## Verifying the merge without the audio

`process_database.py`'s CSV-merging logic was checked against the real,
downloaded Zenodo metadata (not just read from the paper): the resulting
train/dev/test row counts (651/156/192), the per-split valence and
arousal distributions, and the per-split speaker/gender counts (80
development speakers, 40F/40M; 20 test speakers, 10F/10M) all match the
official statistics published on the Zenodo page exactly. What is *not*
verified end-to-end is audio loading and feature extraction, since the
restricted-access audio could not be obtained for this change -- if a
file fails to load, please open an issue with the error.

## License

The metadata is released under CC BY 4.0. The raw audio requires a signed
agreement (research use only; see `EMOVOME_agreement.pdf` in the Zenodo
zip) and may not be redistributed.
