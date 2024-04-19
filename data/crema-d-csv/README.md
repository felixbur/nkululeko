This folder is to import the Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D)
database to nkululeko. Labels are: 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'.

We used the version downloadable from [github](https://github.com/CheyneyComputerScience/CREMA-D) 
(downloaded April 27th 2023).

Download and unzip or git clone the archive.

We only used the [AudioWAV subdirectory](https://www.kaggle.com/datasets/ejlok1/cremad?resource=download). 
You can also download from Kaggle link above; it is smaller than the original Github repo.


```bash
$ cp ~/Download/archive.zip .
$ unzip archive.zip
$ python3 process_database.py
```

then, as a test, you might do

```bash
python -m nkululeko.nkululeko --config data/crema-d/exp_crema-d_audmodel_xgb.ini
```
