# Nkululeko pre-processing for KBES dataset [2]  

KBES is KUET Bangla Emotional Speech (KBES) dataset contains a unique collection of Bangla audio speech for realistic Bangla Speech Emotion Recognition (SER) [2]. To run the pre-processing for Nkululeko, follow the steps below.
Download the dataset from [1], place it in this directory or somewhere else and
ane make a soft link here (`ln -sf`).

```bash
# unzipt the dataset
unzip "KUET Bangla Emotional Speech (KBES) Dataset.zip"
python3 process_database.py
cd ../..
python3 -m nkululeko.resample --config data/kbes/exp.ini
python3 -m nkululeko.nkululeko --config data/kbes/exp.ini
```

Reference:  
[1] <https://data.mendeley.com/datasets/vsn37ps3rx/4>
[2] Billah, M. M., Sarker, M. L., & Akhand, M. A. H. (2023). KBES: A dataset for realistic Bangla speech emotion recognition with intensity level. Data in Brief, 51, 109741. <https://doi.org/10.1016/j.dib.2023.109741>
