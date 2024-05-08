This folder is to import the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
database to nkululeko.

I used the version downloadable from [Zenodo](https://zenodo.org/record/1188976)

Download and unzip the file Audio_Speech_Actors_01-24.zip
```bash
# download original dataset in 48k
$ wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
$ unzip Audio_Speech_Actors_01-24.zip
```

Or, if you prefer the dataset in 16k, you can download from this link:
https://zenodo.org/records/11063852/files/Audio_Speech_Actors_01-24_16k.zip  

run the file
```bash
python3 process_database.py
```

Change to Nkululeko parent directory,

```bash
cd ../..
```

then, as a test, you might do

```bash
python3 -m nkululeko.nkululeko --config data/ravdess/exp_ravdess_os_xgb.ini 
```

Check the results in the results folder under Nkululeko parent directory.