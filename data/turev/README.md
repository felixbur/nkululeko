# Nkululeko pre-processing for TurEV-DB  

```bash
git clone https://github.com/Xeonen/TurEV-DB.git
python3 process_database.py
cd ../..
python3 -m nkululeko.resample --config data/turev/exp.ini
python3 -m nkululeko.nkululeko --config data/turev/exp.ini
```