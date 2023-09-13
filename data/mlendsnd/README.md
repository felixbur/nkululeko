# Nkululeko pre-processin for MLEndSND dataset (public)


Download link:  https://www.kaggle.com/datasets/jesusrequena/mlend-spoken-numerals?resource=download

```bash
$ unzip archive.zip 
$ mv MLEndSND_*.csv MLEndSND_Public
$ python3 process_database.py
$ python3 -m nkululeko.resample --config data/mlendsnd/exp.ini
$ python3 -m nkululeko.nkululeko --config data/mlendsnd/exp.ini
```

Reference:   
[1] https://www.kaggle.com/datasets/jesusrequena/mlend-spoken-numerals