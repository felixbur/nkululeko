# CLSE — Cognitive Load with Speech and EGG

Nkululeko import for the **CLSE** dataset used in the
[INTERSPEECH 2014 ComParE cognitive load sub-challenge](https://www.isca-archive.org/interspeech_2014/schuller14_interspeech.pdf).

- **26 speakers** (20 male, 6 female), native Australian English
- **Three cognitive load levels:** L1=low, L2=medium, L3=high
  (`objective_cognitive_load` 1/2/3)
- **Subjective load rating:** 1–9 Likert scale (`subjective_cognitive_load`)
- **Speech tasks:** readingspanSentence, stroopdualtask, strooptimepressure,
  readingspanLetter, storyreading (UBM)
- **Splits:** train / dev / test (ComParE 2014 partitioning)

Reference:
> Yap et al. (2015). Voice source under cognitive load: Effects and classification.
> *Speech Communication*, 72, 115–126. https://doi.org/10.1016/j.specom.2015.05.007


## Setup

Requires the raw ComParE 2014 distribution (the `extracted_and_merged` folder from
the NAS at `database/DBS-Public/ComParE2014-cognitive-load`) and `silero-vad` for
VAD segmentation:

```bash
pip install silero-vad soundfile pandas
python data/clse/process_database.py /path/to/extracted_and_merged
# output: clse_train.csv (in current dir), clse_dev.csv, clse_test.csv, clse_all.csv
```

To write output files to a specific directory:

```bash
python data/clse/process_database.py /path/to/extracted_and_merged --out_dir data/clse
```


## Using the CSVs in nkululeko

```ini
[DATA]
databases = ['clse_train', 'clse_dev']
clse_train = data/clse/clse_train.csv
clse_train.type = csv
clse_train.split_strategy = train
clse_dev = data/clse/clse_dev.csv
clse_dev.type = csv
clse_dev.split_strategy = dev
target = objective_cognitive_load
```

For regression with native labels mapped to [0, 1]:
```ini
target = objective_cognitive_load
# map 1/2/3 → 0.0/0.5/1.0 via a pre-processing step or nkululeko colnames
```

See `exp.ini` for a complete classification example.


## Label notes

- `objective_cognitive_load`: 1=low, 2=medium, 3=high — the ComParE challenge target.
- `subjective_cognitive_load`: 1–9 self-reported mental effort; NaN for UBM (story-reading) segments.
- Test partition has labels for most files (from the ARFF file released with the challenge);
  a small number of test files (`readingspanLetter`) were excluded from the challenge evaluation
  and have NaN labels.
