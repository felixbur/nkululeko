# Nkululeko pre-processing for Ekorpus dataset (public)  

The Estonian Emotional Speech Corps (EEKK) is a corps created at the Estonian Language Institute within the framework of the state program "Estonian Language Technological Support 2006-2010", which contains read sentences of anger, joy and sadness as well as neutral sentences.The corpus contains 1,234 Estonian sentences that express anger, joy and sadness, or are neutral. Female voice, 44.1 KHz, 16Bit, Mono; wav, textgrid: phonemes, words, sentences.

Download link: https://dagshub.com/kingabzpro/Estonian-Emotional-Speech-Corpus
Alternatively, you can download the file from the github page:


```bash
$ unzip ekorpus.zip
$ python3 process_database.py
$ python3 -m nkululeko.resample --config data/ekorpus/exp.ini
$ python3 -m nkululeko.nkululeko --config data/ekorpus/exp.ini
```


References:  
[1] Altrov, Rene; Pajupuu, Hille 2012. Estonian Emotional Speech Corpus: Theoretical base and implementation. In: 4th International Workshop on Corpora for Research on Emotion Sentiment & Social Signals (ES3), Devillers, L., Schuller, B., Batliner, A., Rosso, P., Douglas-Cowie, E., Cowie, R., Pelachaud, C.(eds.),50-53. Istanbul.

