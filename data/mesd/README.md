# Nkululeko pre-processing for MESD dataset (Public)
The Mexican Emotional Speech Database (MESD) provides single-word utterances for anger, disgust, fear, happiness, neutral, and sadness affective prosodies with Mexican cultural shaping. The MESD has been uttered by both adult and child non-professional actors: 3 female, 2 male, and 6 child voices are available (female mean age ± SD = 23.33 ± 1.53, male mean age ± SD = 24 ± 1.41, and children mean age ± SD = 9.83 ± 1.17). Words for emotional and neutral utterances come from two corpora: (corpus A) composed of nouns and adjectives that are repeated across emotional prosodies and types of voice (female, male, child), and (corpus B) which consists of words controlled for age-of-acquisition, frequency of use, familiarity, concreteness, valence, arousal, and discrete emotion dimensionality ratings. 

The dataset is also available at [2].

```bash
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/mesd/exp.ini
```


References: 
[1] M. M. Duville, L. M. Alonso-Valerdi, and D. Ibarra-Zarate, “The Mexican Emotional Speech Database (MESD): elaboration and assessment based on machine learning,” 43rd Annual International Conference of the IEEE Engineering in Medicine and Biology Society, p. 4, 2021.  
[2] https://github.com/bagustris/multilingual_speech_emotion_dataset  