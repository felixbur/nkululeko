# Nkululeko pre-processing for EAED dataset

## Dataset description

AED is an Egyptian-Arabic emotional speech dataset containing 3,614 audio files [1]. The dataset is a semi-natural one as it was collected from five well-known Egyptian TV series. Each audio file ranged in length from 1 to 8 seconds depending on the completion time of the given sentence. The dataset contains six different emotions: happy, sad, angry, neutral, surprised, and fearful. All audio files were recorded using the open source Audacity Software at sampling rate 44.1KHz. Four different human labelers were assigned to hear the recorded audio files in order to annotate/label them. Then, a fifth labeler was assigned for the task of tie-breaking. The number of speakers in the dataset is 79 including 37 males and 42 females.

Instructions:
The audio files for each series are grouped in a separate folder. Each folder consists of multiple folders, one for each actor/actress in the series. In each actor/actress folder, the audio files are named in the following convention: AA_BB_ CC.wav

AA : Actor unique ID  
BB : the emotion label  
CC : unique number inside this folder

Example: `NellyKarim_happy_ 01.wav` is a file in a folder that belongs to an actress whose name is Nelly Karim and the emotion being conveyed is happy.

## Pre-processing command

First, download the dataset from the link provided in the reference [2]. There are all 27 parts of ZIP files. Extract one file after downloading all files; there will be a single directory from all ZIP files. Then, extract the dataset and run the following command to pre-process the dataset.

```bash
python process_database.py
cd ..
python3 -m nkululeko.resample --config data/eaed/exp.ini
python3 -m nkululeko.nkululeko --config data/eaed/exp.ini
```

Reference:  
[1]  Safwat, S., Salem, M. A.-M., & Sharaf, N. (2024). Building an Egyptian-Arabic Speech Corpus for Emotion Analysis Using Deep Learning. In F. Liu, A. A. Sadanandan, D. N. Pham, P. Mursanto, & D. Lukose (Eds.), PRICAI 2023: Trends in Artificial Intelligence (pp. 320–332). Springer Nature Singapore.  
[2] <https://ieee-dataport.org/documents/egyptian-arabic-emotional-dataset-eaed>
