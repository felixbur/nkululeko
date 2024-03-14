# Nkululeko pre-processing for VIVAE corpus (public)

The Variably Intense Vocalizations of Affect and Emotion Corpus (VIVAE) consists of a set of human non-speech emotion vocalizations. The full set, comprising 1085 audio files, features eleven speakers expressing three positive (achievement/ triumph, sexual pleasure, and surprise) and three negative (anger, fear, physical pain) affective states, each parametrically varied from low to peak emotion intensity. The smaller core set of 480 files represents a fully crossed subsample of the full set (6 emotions x 4 intensities x 10 speakers x 2 items) selected based on judged authenticity. 


```bash
$ wget https://zenodo.org/record/4066235/files/VIVAE.zip
$ unzip VIVAE.zip
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.resample --config data/vivae/exp.ini
$ python3 -m nkululeko.nkululeko --config data/vivae/exp.ini
```

Reference:  
[1] Holz, N., Larrouy-Maestri, P., & Poeppel, D. (2022). The Variably Intense Vocalizations of Affect and Emotion (VIVAE) corpus prompts new perspective on nonspeech perception. Emotion, 22(1), 213–225. http://dx.doi.org/10.1037/emo0001048  
[2] https://zenodo.org/record/4066235