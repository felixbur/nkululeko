# Nkululeko pre-processing for EMOVO dataset (Public)

This article describes the first emotional corpus, named EMOVO, applicable to Italian language,. It is a database built from the voices of up to 6 actors who played 14 sentences simulating 6 emotional states (disgust, fear, anger, joy, surprise, sadness) plus the neutral state. These emotions are the well-known Big Six found in most of the literature related to emotional speech. The recordings were made with professional equipment in the Fondazione Ugo Bordoni laboratories. The paper also describes a subjective validation test of the corpus, based on emotion-discrimination of two sentences carried out by two different groups of 24 listeners. The test was successful because it yielded an overall recognition accuracy of 80%. It is observed that emotions less easy to recognize are joy and disgust, whereas the most easy to detect are anger, sadness and the neutral state.

Download link: https://drive.google.com/file/d/1SUtaKeA-LYnKaD3qv87Y5wYgihJiNJAo/view  

```bash
$ unzip EMOVO.zip
$ python3 process_database.py
# you may need to change permission for resampling
$ chmod -R u+w EMOVO
$ python3 -m nkululeko.resample --config data/emovo/exp.ini
$ python3 -m nkululeko.nkululeko --config data/emovo/exp.ini
```


Reference:  
[1] https://aclanthology.org/L14-1478/  
[2] http://voice.fub.it/activities/corpora/emovo/index.html 
