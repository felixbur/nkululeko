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

Besides [nkululeko.nkululeko], there are other functionalities. The complet functions are:

-   **nkululeko.nkululeko**: doing experiments
-   **nkululeko.predict**: unified prediction module. Predicts labels for
    one or more audio files (`--file`), a folder of audio (`--folder`),
    a CSV list (`--list`) or microphone input (`--mic`), using either a
    feature extractor / autopredict target (`--type feats`, the default)
    or the best model from a trained experiment (`--type model`).
    Replaces the former `nkululeko.demo`, `nkululeko.feature_demo` and
    `nkululeko.testing` modules.
-   **nkululeko.explore**: perform data exploration
-   **nkululeko.augment**: augment the current training data
-   **nkululeko.ensemble**: ensemble a series of models

See the CLI References (Modules) and [API documentation](ini_file.md) for more details.