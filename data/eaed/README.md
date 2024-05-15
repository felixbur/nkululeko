# Nkululeko pre-processing for EAED dataset

## Dataset description

AED is an Egyptian-Arabic emotional speech dataset containing 3,614 audio files. The dataset is a semi-natural one as it was collected from five well-known Egyptian TV series. Each audio file ranged in length from 1 to 8 seconds depending on the completion time of the given sentence. The dataset contains six different emotions: happy, sad, angry, neutral, surprised, and fearful. All audio files were recorded using the open source Audacity Software at sampling rate 44.1KHz. Four different human labelers were assigned to hear the recorded audio files in order to annotate/label them. Then, a fifth labeler was assigned for the task of tie-breaking. The number of speakers in the dataset is 79 including 37 males and 42 females.

Instructions:
The audio files for each series are grouped in a separate folder. Each folder consists of multiple folders, one for each actor/actress in the series. In each actor/actress folder, the audio files are named in the following convention: AA_BB_ CC.wav

AA : Actor unique ID

BB : the emotion label

CC : unique number inside this folder

Example: NellyKarim_happy_ 01.wav is a file in a folder that belongs to an actress whose name is Nelly Karim and the emotion being conveyed is happy.

## Pre-processing command

```bash
```

Reference:  
[1]  
