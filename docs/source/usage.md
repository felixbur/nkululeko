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
-   **nkululeko.demo**: demo the current best model on command line
-   **nkululeko.test**: predict a series of files with the current
    best model
-   **nkululeko.explore**: perform data exploration
-   **nkululeko.augment**: augment the current training data
-   **nkululeko.predict**: predict a series of files with a given model
-   **nkululeko.ensemble**: ensemble a series of models

See the CLI References (Modules) and [API documentation](ini_file.md) for more details.