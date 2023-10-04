# Nkululeko pre-processing for CLAC dataset (Public)


```bash
$ wget https://data.csail.mit.edu/placesaudio/CLAC-Dataset.zip
# if your download is interrupted, you can resume it with the following command
$ wget -c https://data.csail.mit.edu/placesaudio/CLAC-Dataset.zip
$ unzip CLAC-Dataset.zip
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.resample --config data/clac/exp.ini
$ python3 -m nkululeko.nkululeko --config data/clac/exp.ini
```

Reference:  
[1] Haulcy, R., & Glass, J. (2021). Clac: A speech corpus of healthy english speakers. Proceedings of the Annual Conference of the International Speech Communication Association, INTERSPEECH, 1, 201–205. https://doi.org/10.21437/Interspeech.2021-1810  