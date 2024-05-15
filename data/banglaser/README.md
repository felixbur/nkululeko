# Nkululeko pre-processing for Bangla SER dataset

Download link: <https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/t9h6p943xy-5.zip>
Filename convention:  

AA-BB-CC-DD-EE-FF-GG.wav

```bash
wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/t9h6p943xy-5.zip
unzip t9h6p943xy-5.zip
python3 process_database.py
cd ../..
python3 -m nkululeko.resample --config data/banglaser/exp.ini
python3 -m nkululeko.nkululeko --config data/banglaser/exp.ini

```  

Reference:  
[1]
