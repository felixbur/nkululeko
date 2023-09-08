# Nkululeko preprocssing for ESD dataset

The data should process directory "ESD" in current working directory as default data directory input. The output is two CSV file esd_train.csv and esd_test.csv containing "file", "emotion", "language", and "gender".

Download dataset from here: https://drive.google.com/file/d/1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v/view, 
and extract it to "ESD" directory.

```bash
$ unzip Emotional Speech Dataset (ESD).zip
$ mv 'Emotion Speech Dataset' ESD
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/esd/exp.ini 
```



Reference:  
[1] 

