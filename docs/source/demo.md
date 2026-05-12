# Predicting with a Trained Model

> The former `nkululeko.demo` module has been merged into the unified
> `nkululeko.predict` module. This page shows how to reproduce the old demo
> workflow — testing a previously trained model with an audio file, a list of
> files or live microphone input — using `nkululeko.predict --type model`.
> See [predict.md](predict.md) for the full reference.

## Concept refresher

- **train** is used to train a supervised model.
- **dev** is used to evaluate the model during development (early stopping,
  hyperparameter tuning).
- **test** is the held-out set that should only be used at the very end.

Classical models (SVM, XGB) usually only need train and test. Neural-network
models (MLP, CNN) typically use train, dev and test.

## Configuration

Before predicting with a saved model you need to have trained it once. Add the
following to your INI:

```ini
[EXP]
save = True

[MODEL]
save = True
```

Then train the experiment as usual:

```bash
python -m nkululeko.nkululeko --config myconfig.ini
```

## Predict a single file

```bash
python -m nkululeko.predict \
    --config data/ravdess/exp1.ini \
    --type model \
    --file data/ravdess/Actor_01/03-01-01-01-01-01-01.wav
```

The output (printed to stdout and written to
`data/ravdess/Actor_01/03-01-01-01-01-01-01_result.txt`):

```
DEBUG: predict: nkululeko 1.6.3: unified predict
data/ravdess/...wav   angry: 0.314
data/ravdess/...wav   happy: 0.312
data/ravdess/...wav   neutral: 0.042
data/ravdess/...wav   sad: 0.332
data/ravdess/...wav   predicted: sad
DEBUG: predict: DONE
```

## Predict a list of files

To process many files and write a single CSV:

```bash
python -m nkululeko.predict \
    --config data/ravdess/exp1.ini \
    --type model \
    --list data/ravdess/ravdess_test.csv \
    --outfile /tmp/ravdess_test_predict.csv
```

The original columns of `ravdess_test.csv` are preserved in the output. For a
classification model the additional columns are one score column per class
plus `predicted`. For a regression model a single `predicted` column is added.

Example list CSV (audformat, filewise index):

```text
file
./Actor_21/03-01-07-01-01-01-21.wav
./Actor_21/03-01-06-01-02-02-21.wav
./Actor_21/03-01-06-02-01-02-21.wav
```

Example output CSV:

```text
file,start,end,angry,happy,neutral,sad,predicted
./Actor_21/03-01-07-01-01-01-21.wav,0 days,,0.314,0.315,0.038,0.332,sad
./Actor_21/03-01-06-01-02-02-21.wav,0 days,,0.314,0.313,0.041,0.332,sad
./Actor_21/03-01-06-02-01-02-21.wav,0 days,,0.314,0.316,0.037,0.332,sad
```

## Predict every audio file in a folder

```bash
python -m nkululeko.predict \
    --config data/ravdess/exp1.ini \
    --type model \
    --folder ./Actor_21 \
    --outfile /tmp/actor21_predict.csv
```

## Live microphone

```bash
python -m nkululeko.predict --config data/ravdess/exp1.ini --type model --mic
```

The microphone loop records `5` seconds at `16 kHz`, runs the model and prints
the per-class scores to stdout. Press *q* + *Enter* to exit.

## Related

- [predict.md](predict.md) — full reference for `nkululeko.predict`.
- [test_module.md](test_module.md) — using the predict module to re-evaluate
  on labeled test data.
- Background blog post:
  <http://blog.syntheticspeech.de/2022/01/24/nkululeko-try-out-demo-a-trained-model/>
