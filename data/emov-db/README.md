# Nkululeko pre-processing for EMOV-DB dataset (public)

he Emotional Voices Database: Towards Controlling the Emotional Expressiveness in Voice Generation Systems

This dataset is built for the purpose of emotional speech synthesis. The transcript were based on the CMU arctic database: http://www.festvox.org/cmu_arctic/cmuarctic.data. It includes recordings for four speakers- two males and two females. The emotional styles are neutral, sleepiness, anger, disgust and amused. Each audio file is recorded in 16bits .wav format.  

- Spk-Je (Female, English: Neutral(417 files), Amused(222 files), Angry(523 files), Sleepy(466 files), Disgust(189 files))

- Spk-Bea (Female, English: Neutral(373 files), Amused(309 files), Angry(317 files), Sleepy(520 files), Disgust(347 files))

- Spk-Sa (Male, English: Neutral(493 files), Amused(501 files), Angry(468 files), Sleepy(495 files), Disgust(497 files))

- Spk-Jsh (Male, English: Neutral(302 files), Amused(298 files), Sleepy(263 files))

File naming (audio_folder): anger_1-28_0011.wav 
- first word (emotion style) 
- 1-28 - annotation doc file range
- Last four digit is the sentence number.

File naming (annotation_folder): anger_1-28.TextGrid 
- first word (emotional style)
- 1-28- annotation doc range

Runnig the code:

```bash
$ python3 process_database.py
$ python3 -m nkululeko.resample --config data/emov-db/exp.ini
$ python3 -m nkululeko.nkululeko --config data/emov-db/exp.ini
```

References:  
[1] http://www.openslr.org/115/  
[2] Adigwe, A., Tits, N., Haddad, K. El, Ostadabbas, S., & Dutoit, T. (2018). The Emotional Voices Database: Towards Controlling the Emotion Dimension in Voice Generation Systems. SLSP. http://arxiv.org/abs/1806.09514