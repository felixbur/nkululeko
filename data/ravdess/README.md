This folder is to import the 
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
database to nkululeko.

I used the version downloadable from [Zenodo](https://zenodo.org/record/1188976)

Download and unzip the file Audio_Speech_Actors_01-24.zip

run the file
```
python process_database.py
```

then, as a test, you might do
```
python -m nkululeko.nkululeko --config exp_ravdess_os_xgb.ini 
```