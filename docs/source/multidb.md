# Multidb module for database comparison

With nkululeko since version 0.77.7 there is a new interface named multidb, which lets you compare several databases.

You can state their names in the [EXP] section and they will then be processed one after each other and against each other; the results are stored in a file called heatmap.png in the experiment folder.

>> YOU NEED TO OMIT THE PROJECT NAME!

Here is an example of such an INI file

```ini
[EXP]
root = ./experiments/emodbs/
#  DON'T give it a name, 
# this will be the combination 
# of the two databases: 
# traindb_vs_testdb
epochs = 1
databases = ['emodb', 'polish']
[DATA]
root_folders = ./experiments/emodbs/data_roots.ini
target = emotion
labels = ['neutral', 'happy', 'sad', 'angry']
[FEATS]
type = ['os']
[MODEL]
type = xgb
```
You can (but don't have to) state the specific dataset values in an external file like above.

```ini
[DATA]
emodb = ./data/emodb/emodb
emodb.split_strategy = specified
emodb.test_tables = ['emotion.categories.test.gold_standard']
emodb.train_tables = ['emotion.categories.train.gold_standard']
emodb.mapping = {'anger':'angry', 'happiness':'happy', 'sadness':'sad', 'neutral':'neutral'}
polish = ./data/polish_emo
polish.mapping = {'anger':'angry', 'joy':'happy', 'sadness':'sad', 'neutral':'neutral'}
polish.split_strategy = speaker_split
polish.test_size = 30
```

Finally, you can run the experiment with the following command:

```bash
python -m nkululeko.multidb --config my_conf.ini
```

Here's a result with two databases.

![heatmap](./images/heatmap-multidb.png)

Source: http://blog.syntheticspeech.de/2024/01/02/nkululeko-compare-several-databases/