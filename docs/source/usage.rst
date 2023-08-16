Usage
=====

The main usage of Nkululeko is as follows:

.. code-block:: bash

    python -m nkululeko.nkululeko --config INI_FILE.ini

where `INI_FILE.ini` is a configuration file. The only file needed by the user is the INI file (after preparing the dataset). That's why we said this tool is intented without or less coding. The example of configuration file (INI_FILE.ini) is given below. See `INI file <ini.html>`__ for complete options.

.. code-block:: ini

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


Besides `nkululeko.nkululeko`, there are other functionalities. The completen functions are:
    * **nkululeko.nkululeko**: doing experiments
    * **nkululeko.demo**: demo the current best model on command line
    * **nkululeko.test**: predict a series of files with the current best model
    * **nkululeko.explore**: perform data exploration
    * **nkululeko.augment**: augment the current training data
    * **nkululeko.predict**: predict a series of files with a given model

.. _Google Colab: https://colab.research.google.com/drive/1GYNBd5cdZQ1QC3Jm58qoeMaJg3UuPhjw?usp=sharing#scrollTo=4G_SjuF9xeQf'
.. _Kaggle: https://www.kaggle.com/felixburk/nkululeko-hello-world-example