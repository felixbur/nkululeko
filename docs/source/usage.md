# Usage

The main usage of Nkululeko is as follows:

```bash
python -m nkululeko.[MODULE] --config [CONFIG_FILE.ini]
# Example to run the experiment
python -m nkululeko.nkululeko --config INI_FILE.ini
```

where [INI\_FILE.ini](ini_file.md) is a configuration file. The only file
needed by the user is the INI file (after preparing the dataset).
That\'s why we said this tool is intended without or less coding. The
example of configuration file (INI\_FILE.ini) is given below. See [INI
file](ini_file.md) for complete options.

``` {ini}
[EXP]
root = ./results
name = exp_emodb
[DATA]
databases = ['emodb']
emodb = ./emodb/
emodb.split_strategy = specified
emodb.train_tables = ['emotion.categories.train.gold_standard']
emodb.test_tables = ['emotion.categories.test.gold_standard']
target = emotion
labels = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
[FEATS]
type = ['os']
[MODEL]
type = xgb
```

Besides [nkululeko.nkululeko], there are other functionalities. The complete functions are:

-   **nkululeko.nkululeko**: run experiments — trains a model and evaluates
    it.  When `DATA.tests` is set in the INI file **and** a saved
    experiment already exists on disk, training is skipped automatically
    and the stored best model is evaluated on the new test database instead
    (produces a confusion matrix, a text report, and a predictions CSV).
    See [test_new_database.md](test_new_database.md).
-   **nkululeko.predict**: unified prediction module. Predicts labels for
    one or more audio files (`--file`), a folder of audio (`--folder`),
    a CSV list (`--list`) or microphone input (`--mic`), using either a
    feature extractor / autopredict target (`--type feats`, the default)
    or the best model from a trained experiment (`--type model`).
    Autopredict targets include `age`, `gender`, `emotion`, `arousal`,
    `valence`, `dominance`, `mos`, `snr`, `pesq`, `sdr`, `stoi`, `text`,
    `textclassification`, and `translation`.
    Replaces the former `nkululeko.demo`, `nkululeko.feature_demo` and
    `nkululeko.testing` modules.  See [predict.md](predict.md).
-   **nkululeko.explore**: perform data exploration
-   **nkululeko.augment**: augment the current training data
-   **nkululeko.ensemble**: ensemble a series of models

See the CLI References (Modules) and [API documentation](ini_file.md) for more details.