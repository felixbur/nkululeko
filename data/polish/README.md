# Nkululeko pre-processing for Polish dataset

## Dataset description  

The data set presents emotions recorded in sound files that are expressions of Polish speech. Statements were made by people aged 21-23, young voices of 5 men. Each person said the following words / nie – no, oddaj - give back, podaj – pass, stop - stop, tak - yes, trzymaj -hold / five times representing a specific emotion - one of three - anger (a), fear (s), neutral emotion (n).

Thus, for a speaker, 90 audio files (1 speaker *6 words* 5 repetitions * 3 emotions = 90 files) were recorded, 30 files for each emotion per speaker. There were 5 speakers, so there are 450 files. The file marks are as follows: for example, the file /m1.a_nie_2.wav/, which means / m1 / - the first speaker, / a / -aggression, / nie / - the spoken word, / 2 / - the second repetition of a given word.

## Pre-processing command

Download link: <https://mostwiedzy.pl/en/open-research-data/emotions-in-polish-speech-recordings,11190523461146169-0/download>

```bash
mkdir POLISH
# if downlading from original dataset
unzip Speech_emotions.zip -d POLISH
# if using the resampled version
unzip polish_speech_emotions.zip
python3 process_database.py
cd ../..
python3 -m nkululeko.resample --config data/polish/exp.ini
python3 -m nkululeko.nkululeko --config data/polish/exp.ini
```

Reference:  
[1] Mięsikowska, M., & Świsulski, D. (2020). Emotions in Polish speech recordings  [dataset]. Gdańsk University of Technology. <https://doi.org/10.34808/h46c-hb44>  
[2] <https://mostwiedzy.pl/en/open-research-data/emotions-in-polish-speech-recordings,11190523461146169-0>
