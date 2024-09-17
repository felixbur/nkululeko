
Nkululeko database repository
=============================

# Data


This is the default top directory for Nkululeko data import.Each database should be in its own subfolder (you can also use `ln -sf`` to soft link original database path to these subfolders) and contain a README how to import the data to Nkululeko CSV or audformat.
## Accessibility


The column `access` in the table below indicates the database's accessability. The following values are used:
- `public`: the database is publicly available in the internet and can be downloaded directly without any restrictions.
- `restricted`: the database is publicly available on the internet but requires registration or other restrictions to download.
- `private`: the database is not publicly available on the internet and requires the private information of the owner of the dataset.


To support open science and reproducible research, we only accept PR and recipes for public dataset for now on.
## Databases

|Name|Target|Description|Access|
| :---: | :---: | :---: | :---: |
|emorynlp|emotion|English,FromFriendsTV|public|
|emns|emotion,intensity|British,singlespeaker,UAR=0.479|public|
|test|none|Test data for nkululeko|public|
|catsvsdogs|cats_dogs|kaggle test set|public|
|clac|healthyspeech,age,gender|English|public|
|vivae|emotion|Englishvocalbursts|public|
|emofilm|emotion|English,Spanish,Italian|restricted|
|emozionalmente|emotion|Italian|public|
|laughter-types|laughter|MasterThesisfromTUBerlin|public|
|savee|emotion|English,fromtfds|restricted|
|emovo|emotion|Italian|restricted|
|subesco|emotion|Bangla|public|
|oreau2|emotion|French|public|
|mess|emotion|English|public|
|emov-db|emotion|English|public|
|odyssey-cat-2024|emotion|Data for Odyssey 2024 challenge, needs MSPPodcast|restricted|
|ravdess|emotion,speaker|English|public|
|erysac|emotion|Russian,children|public|
|demos|emotion|Italian|restricted|
|emodb|emotion|German|public|
|jnv|emotion|Japanese,non-verbals|public|
|aesdd|emotion|amhariclanguage|public|
|msp-podcast|emotion,VAD|English|restricted|
|baved|emotion|Arabic|public|
|asvp-esd|emotion|Multilingual,alsocontainvocalbursts|public|
|ekorpus|emotion|Estonian|public|
|iemocap|emotion,VAD|English|restriced|
|jl|emotion|English|public|
|syntact|emotion|SynthesizedGermanspeech|public|
|kia|emotion|Korean,wake-upword|public|
|portuguese|emotion|Portuguese|public|
|crema-d|emotion|English,adoptedfromtfds|public|
|thorsten-emotional|emotion|German|public|
|jvnv|emotion|Japanese,verbalandnon-verbal|public|
|mesd|emotion|Mexican|public|
|enterface|emotion|Multilingual|public|
|turev|emotion|Turkish|public|
|nEMO|emotion,VAD|Polish|public|
|eaed|emotion|Arabic|public|
|ased|emotion|Greek|public|
|cafe|emotion|Childrenspeech,CanadianFrench|public|
|banglaser|emotion|Bengali|public|
|mlendsnd|emotion|English|public|
|gerparas|valence,arousal,dominance|German|restricted|
|androids|depression|English|public|
|kbes|emotion|Bengali|public|
|tess|emotion|BritishEnglish(Toronto)|public|
|jtes|emotion|Japanese|private|
|meld|emotion|English,FromFriendsTV|public|
|urdu|emotion|Urdu|public|
|polish|emotion|Polish|public|
|cmu-mosei|sentiment,emotion|English,originallinkdead|public|
|SVD|pathologicalspeech|German|public|
|msp-improv|emotion,VAD,naturalness|English|restricted|
|shemo|emotion|Persian|public|
|esd|emotion|English,Chinese|public|

## Performance
