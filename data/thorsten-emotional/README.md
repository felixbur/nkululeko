# Nkulueko preprocessing for Thorsten-Emotional dataset (public)

This workflow used the OpenSLR version of the dataset.

```bash
$ wget https://www.openslr.org/resources/110/thorsten-emotional_v02.tgz
$ unzip thorsten-emotional_v02.tgz
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/thorsten-emotional/exp.ini
```


Reference:  
[1] https://www.openslr.org/110/
[2] https://zenodo.org/record/5525023
