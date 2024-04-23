This folder is to import the Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D) database to nkululeko. 

Labels are: 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'.

Based on the [audb version](https://github.com/audeering/crema-d)

Load the database with python:
```bash
$ python load_db.py
```
**inside** the crema-d folder!

then, as a test, you might do

```bash
python -m nkululeko.nkululeko --config data/crema-d/test_age.ini
python -m nkululeko.nkululeko --config data/crema-d/test_emotion.ini
```
