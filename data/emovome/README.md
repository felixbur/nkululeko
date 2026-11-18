# Nkululeko pre-processing for the EMOVOME dataset (text)

EMOVOME (Emotional Voice Messages) is a Spanish dataset for emotion
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

This example is **text-only**: it trains on the manually corrected
transcript of each message (via BERT embeddings), not on the voice
recording. That's a deliberate choice, not just a workaround -- the
dataset's own baseline paper found text at least as informative as
speech for these same labels (a fine-tuned multilingual BERT on the
transcripts reached 61% UAR for valence, against 49% for an
eGeMAPS+SVM speech baseline). It also means this example needs **no
restricted-access audio at all**: everything it uses is in the openly
downloadable Zenodo metadata.

Reference:
> L. Gómez-Zaragozá, R. del Amor, M. J. Castro-Bleda, V. Naranjo, M.
> Alcañiz Raya, J. Marín-Morales, "EMOVOME Database: Advancing Emotion
> Recognition in Speech Beyond Staged Scenarios", 2024.
> https://doi.org/10.48550/arXiv.2403.02167

Dataset homepage (Zenodo): https://doi.org/10.5281/zenodo.10694370

## Getting the data

Download and unzip `emovome.zip` (1.3 GB, open access -- the Zenodo
page describes a separate, signed-agreement process for the raw voice
recordings, but this zip itself needs no such agreement) into this
directory (`data/emovome/`), so `labels.csv`, `Baseline_emotion_
recognition/`, `Questionnaires/` etc. end up sitting right here, next
to `process_database.py`:

```bash
cd data/emovome
wget "https://zenodo.org/records/10694370/files/emovome.zip?download=1" -O emovome.zip
unzip emovome.zip
rm emovome.zip
```

That's it -- no audio needed. Then, still from this directory:

```bash
python process_database.py
python -m nkululeko.nkululeko exp.ini
```

`process_database.py` takes no required options: it reads the metadata
(`labels.csv`, `participants_ids.csv`, `transcriptions.csv`,
`Baseline_emotion_recognition/partitions.csv`,
`Questionnaires/sociodemographic_english.csv`) from the current
directory and writes `emovome_train.csv` / `emovome_dev.csv` /
`emovome_test.csv` (the official partition) plus `emovome_all.csv`
(all three, with a `split` column). Pass `--zenodo_dir`/`--out_dir` to
override the input/output locations if your own layout differs.

### Why there's still a "file" column, and a `placeholder_audio/` folder

nkululeko's pipeline is built around audio files: every database needs
a readable audio file per row so it can work out each row's duration,
even for a text-only experiment where that audio is never actually
read for features. `process_database.py` therefore also writes a tiny
(0.5 s) silent `.wav` per row into `placeholder_audio/`, and the `file`
column points there. `exp.ini`'s `FEATS.type = ['bert']` never opens
that file -- it reads the `transcription` column directly -- so the
placeholder's content carries no signal at all; it exists purely to
satisfy nkululeko's bookkeeping.

If you do obtain the real, restricted-access audio (`Audios/Emotions/`,
one `<file_id>.ogg` per row) separately and want a speech-based
experiment instead, point `emovome_*.csv`'s `file` column at those
files (e.g. with a small script) and switch `FEATS.type` to e.g.
`['os']`.

## Output columns

| column | values | source |
|---|---|---|
| `file` | path to a placeholder silent `.wav` -- see above, not the real voice message | generated |
| `speaker` | alphanumeric participant ID | `participants_ids.csv` |
| `gender` | `female` / `male` | `Questionnaires/sociodemographic_english.csv` |
| `age_range` | e.g. `18-25` (bucketed, not a single age) | `Questionnaires/sociodemographic_english.csv` |
| `valence` | `negative` / `neutral` / `positive` | `labels.csv: valence_C` (combined non-expert + expert) |
| `arousal` | `low` / `neutral` / `high` | `labels.csv: arousal_C` |
| `category` | `happiness`/`neutral`/`anger`/`surprise`/`sadness`/`fear`/`disgust` | `labels.csv: category_E` (expert only) |
| `transcription` | manually corrected transcript -- the actual model input | `transcriptions.csv: Manual` |
| `transcription_asr` | raw Amazon Transcribe output (not used by `exp.ini`) | `transcriptions.csv: AWS` |

`exp.ini` targets `valence` (the dataset's main published benchmark); set
`target = arousal` or `target = category` for the other tasks -- update
`labels` accordingly (`category` is imbalanced: 342 happiness / 225
neutral / 199 anger / 118 surprise / 72 sadness / 35 fear / 8 disgust
messages across all 999, so the original paper's category baseline keeps
only happiness/anger/neutral/surprise; adding
`emovome_train.filter = {'category': ['happiness', 'anger', 'neutral', 'surprise']}`
(and the same for `emovome_dev`/`emovome_test`) reproduces that.)

## Verified result

Run end-to-end (real metadata and transcripts, `FEATS.type = ['bert']`
with `bert.model = google-bert/bert-base-multilingual-cased`, `MODEL.type
= mlp`): test UAR **0.62**, closely matching the dataset's own
multilingual-BERT valence baseline (~0.61 UAR). The first run downloads
the BERT model and extracts embeddings for all 999 texts (a few minutes
on CPU); both are cached (model under the usual HuggingFace cache,
embeddings under `results/emovome_valence/store/`), so later runs are
fast.

`process_database.py`'s CSV-merging logic was also checked against the
real, downloaded Zenodo metadata directly: the resulting train/dev/test
row counts (651/156/192), the per-split valence/arousal distributions,
and the per-split speaker/gender counts (80 development speakers,
40F/40M; 20 test speakers, 10F/10M) all match the official statistics
published on the Zenodo page exactly.

## License

The metadata (including the transcripts this example trains on) is
released under CC BY 4.0. The raw audio requires a signed agreement
(research use only; see `EMOVOME_agreement.pdf` in the Zenodo zip) and
may not be redistributed -- irrelevant to this example, but relevant if
you pursue a speech-based one (see above).
