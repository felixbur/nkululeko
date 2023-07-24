Hello world!
------------

Here is an hello world of using Nkululeko. This hello world is also available in `Google Colab`_ and `Kaggle`_.


1. Download the EmoDB dataset.::
    wget https://tubcloud.tu-berlin.de/s/LfkysdXJfiobiEG/download/emodb.zip

2. Unzip the dataset.::
    unzip emodb.zip

3. Create INI file, you can take example from `tests` directori. For more explanation about format of INI FILE, see `INI file <./ini.rst>`__.::
    
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

3. Run the experiment.::
    python -m nkululeko -i emodb.ini

4. Check the results in the `results` directory.::


.. _Google Colab: https://colab.research.google.com/drive/1GYNBd5cdZQ1QC3Jm58qoeMaJg3UuPhjw?usp=sharing#scrollTo=4G_SjuF9xeQf'
.. _Kaggle: https://www.kaggle.com/felixburk/nkululeko-hello-world-example