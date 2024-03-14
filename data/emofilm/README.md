# Nkululeko dataset processing for EmoFilm dataset

EmoFilm dataset can be requested from the following Zenodo link: https://zenodo.org/record/7665999. 
We used the third version of the dataset.

```bash
$ python convert_to_16k.py
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/emofilm/exp_emofilm_audmodel_svm.ini
```

See each python files to inlcude the path to the dataset (default is current directory).