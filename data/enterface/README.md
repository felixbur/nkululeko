# Nkululeko pre-processing for ENTERFACE dataset

This dataset is available at [2].  
When resampling from original AVI files, it needs FFMPEG < 7
(`onda install -c conda-forge 'ffmpeg<7'`)


```bash
$ python3 process_database.py
cd ../..
$ python3 -m nkululeko.resample --config data/enterface/exp_resample.ini
$ python3 -m nkululeko.nkululeko --config data/enterface/exp.ini
```


References:  
[1] Martin, Olivier, Irene Kotsia, Benoit Macq, and Ioannis Pitas. "The eNTERFACE'05 audio-visual emotion database." In 22nd international conference on data engineering workshops (ICDEW'06), pp. 8-8. IEEE, 2006.
[2] https://github.com/bagustris/speech_emotion_dataset