This folder is to import the 
Emozionalmente: a crowdsourced Italian speech emotional corpus
database to nkululeko.

I used the version downloadable from [zenodo](https://zenodo.org/record/6569824)

downloaded June 20th 2023

Download and unzip

run
```
python create.py
```

then, you could run
```
python -m nkululeko.nkululeko --config nkulu_os_xgb.ini
```
to test the database and
```
python -m nkululeko.explore --config nkulu_os_xgb.ini
```
to see the data distribution

Should result into a confusion matrix like this

![alt text](results/images/run_0/data_xgb_os__0_000_cnf.png "Confusion matrix")