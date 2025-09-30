# Nkululeoko pre-processing for UUDB dataset  

## directory structure
```bash
$ tree -d -L 1 UUDB
UUDB
├── datasets
├── doc
├── Sessions
├── tools
└── var

5 directories
```
### Generate CSV metadata files and run experiments
```bash
python process_database.py
cd ../..
python -m nkululeko.nkululeko data/uudb/exp.ini
```

## Reference:  
[1] https://research.nii.ac.jp/src/en/UUDB.html  