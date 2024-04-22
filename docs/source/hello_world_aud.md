# Hello world [AUDFORMAT]

Here is a hello world of using Nkululeko using dataset in [audformat](https://audeering.github.io/audformat/). This hello world is also
available in [Google
Colab](https://colab.research.google.com/drive/1GYNBd5cdZQ1QC3Jm58qoeMaJg3UuPhjw?usp=sharing#scrollTo=4G_SjuF9xeQf')
and
[Kaggle](https://www.kaggle.com/felixburk/nkululeko-hello-world-example).

In this setup, we will use the Berlin Emodb dataset. Check the `data/emodb` directory for the dataset and follow the instructions below.  

Change the directory to the root of the project.

```bash
# Download using wget
$ wget https://zenodo.org/record/7447302/files/emodb.zip
# Unzip
$ unzip emodb.zip
# change to Nkululeko parent directory
$ cd ..
# run the nkululeko experiment
$ python -m nkululeko.nkululeko --config tests/exp_emodb_os_xgb.ini
```

Then, check the results in the `results` directory.

You can experiment with changing some paramaters in INI file. For instance, change the `type` of the model to `xgb` or `svm` and see how the results change.

```ini
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
[PLOT]
```
