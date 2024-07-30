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
...
# sample outputs
DEBUG: reporter: Best score at epoch: 0, UAR: .681, (+-.611/.747), ACC: .672
DEBUG: reporter: labels: ['angry', 'neutral', 'sad', 'happy']
DEBUG: reporter: result per class (F1 score): [0.681, 0.494, 0.886, 0.674] from epoch: 0
DEBUG: experiment: Done, used 180.702 seconds
DONE
```  

Reference:  
[1] Das, R. K., Islam, N., Ahmed, M. R., Islam, S., Shatabda, S., & Islam, A. K. M. M. (2022). BanglaSER: A speech emotion recognition dataset for the Bangla language. Data in Brief, 42, 108091. https://doi.org/10.1016/j.dib.2022.108091
