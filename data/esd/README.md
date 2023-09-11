# Nkululeko preprocssing for ESD dataset

The data should process directory "ESD" in current working directory as default data directory input. The output is two CSV file esd_train.csv and esd_test.csv containing "file", "emotion", "language", "speaker", "speaker_idx", and "gender".

Download dataset from here: https://drive.google.com/file/d/1scuFwqh8s7KIYAfZW1Eu6088ZAK2SI-v/view, 
and extract it to "ESD" directory.

```bash
$ unzip Emotional Speech Dataset (ESD).zip
$ mv 'Emotion Speech Dataset' ESD
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/esd/exp.ini 
```



Reference:  
[1] Zhou, K., Sisman, B., Liu, R., & Li, H. (2021). Seen and Unseen Emotional Style Transfer for Voice Conversion with A New Emotional Speech Dataset. ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021-June, 920–924. https://doi.org/10.1109/ICASSP39728.2021.9413391

