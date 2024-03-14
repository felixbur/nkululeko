# Nkululeko pre-processing for AESDD dataset

The motive for the creation of the database was the absence of a publically available high-quality database for SER in Greek, a realization made during the research on an emotion-triggered lighting framework for theatrical performance. The database utterances with five emotions: anger, disgust, fear, happiness, and sadness.

The first version of the speech emotion recognition dataset was created in collaboration with a group of professional actors, who showed vivid interest in the proposed framework. Dynamic (in AESDD) refers to the intention of constantly expanding the database through the contribution of actors and performers that are involved, or interested in the project. While the call for contribution addresses to actors, the SER models that are trained on the AESDD are not exclusively performance-oriented.

The first version of the AESDD was presented in [1].

Download link: https://mega.nz/#F!0ShVXY7C!-73kVoK05OjTPEA95UUvMw

```bash
# dataset can be located and extracted to other direcoties
$ unzip Acted Emotional Speech Dynamic Database.zip
$ ln -sf Acted Emotional Speech Dynamic Database AESDD 
$ python3 process_database.py
$ python3 -m nkululeko.resample data/aesdd/exp.ini
$ python3 -m nkululeko.nkululeko data/aesdd/exp.ini
```

References:  
[1] Vryzas, N., Kotsakis, R., Liatsou, A., Dimoulas, C. A., & Kalliris, G. (2018). Speech emotion recognition for performance interaction. Journal of the Audio Engineering Society, 66(6), 457-467.  
[2] http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/