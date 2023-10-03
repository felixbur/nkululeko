# Nkululeko pre-processing for MESS dataset (public)

This database of emotional speech has been validated for use in experiments of auditory perception of emotion. Category ratings and emotional dimension ratings of activation and pleasantness are available from the researcher upon request.  

The database can be downloaded from [2] and is also available in [3].

```bash
$ python3 process_database.py 
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/mess/exp.ini
```


References:  
[1] Morgan, S. D. (2019). Categorical and Dimensional Ratings of Emotional Speech: Behavioral Findings From the Morgan Emotional Speech Set. Journal of Speech, Language, and Hearing Research, 62(11), 4015-4029.  
[2] https://zenodo.org/record/7378320  
[3] https://github.com/bagustris/multilingual_speech_emotion_dataset