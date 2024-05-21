# Nkululeko pre-processing for KIA dataset [1]  

```bash
wget https://zenodo.org/records/7091465/files/hi_kia_1.0.tar.gz
tar -xvf hi_kia_1.0.tar.gz
python3 process_database.py
cd ../..
python3 -m nkululeko.resample --config data/kia/exp.ini
python3 -m nkululeko.nkululeko --config data/kia/exp.ini
```

Reference:  
[1]  Kim, T., Doh, S., Lee, G., Jeon, H., Nam, J., & Suk, H. (2022). Hi , KIA : A Speech Emotion Recognition Dataset for Wake-Up Words. November, 1587–1592.
