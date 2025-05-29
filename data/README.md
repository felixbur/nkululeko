Nkululeko database repository
=============================

# Data

This is the default top directory for Nkululeko data import. Each database should be in its own subfolder (you can also use `ln -sf` to soft link original database path to these subfolders) and contain a README how to import the data to Nkululeko CSV or audformat.

## Dataset License Information

Each dataset may have its own license terms. Please refer to the README or license file within each dataset subfolder for specific license details. If not present, consult the official source of the dataset for license and usage information. If you add a new dataset, please include a README with license and citation information.

## Accessibility

The column `access` in the table below indicates the database's accessability. The following values are used:
- `public`: the database is publicly available in the internet and can be downloaded directly without any restrictions.
- `restricted`: the database is publicly available on the internet but requires registration or other restrictions to download.
- `private`: the database is not publicly available on the internet and requires the private information of the owner of the dataset.


To support open science and reproducible research, we encourage to submit PR and recipes for public dataset for now on.
|Name|Target|Description|Access|License|
| :--- | :--- | :--- | :--- | :--- |
|emorynlp|emotion|English Emotion Dataset from Friends TV Show|public|unknown|
|emns|emotion,intensity|British, singles peaker, UAR=.479|public|unknown|
|test|none|Test data for nkululeko|public|unknown|
|clac|healthyspeech,age,gender|English|public|unknown|
|vivae|emotion|English vocal bursts|public|unknown|
|emofilm|emotion|English, Spanish, Italian|restricted|unknown|
|emozionalmente|emotion|Italian|public|unknown|
|laughter-types|laughter|Master Thesis from TUBerlin|public|unknown|
|savee|emotion|English, from tfds|restricted|unknown|
|emovo|emotion|Italian|restricted|unknown|
|subesco|emotion|Bangla|public|unknown|
|oreau2|emotion|French|public|unknown|
|mess|emotion|English|public|unknown|
|emov-db|emotion|English|public|unknown|
|odyssey-cat-2024|emotion|Data for Odyssey 2024 challenge, needs MSPPodcast|restricted|unknown|
|ravdess|emotion,speaker|English|public|unknown|
|erysac|emotion|Russian, children|public|unknown|
|demos|emotion|Italian|restricted|unknown|
|emodb|emotion|German|public|unknown|
|jnv|emotion|Japanese, non-verbals|public|unknown|
|aesdd|emotion|amharic language|public|unknown|
|msp-podcast|emotion,VAD|English|restricted|unknown|
|baved|emotion|Arabic|public|unknown|
|asvp-esd|emotion|Multilingual, also contain vocal bursts|public|unknown|
|ekorpus|emotion|Estonian|public|unknown|
|iemocap|emotion,VAD|English|restriced|unknown|
|jl|emotion|English|public|unknown|
|syntact|emotion|Synthesized German speech|public|unknown|
|kia|emotion|Korean, wake-up word|public|unknown|
|portuguese|emotion|Portuguese|public|unknown|
|crema-d|emotion|English,adopted from tfds|public|unknown|
|thorsten-emotional|emotion|German|public|unknown|
|jvnv|emotion|Japanese, verbal and non-verbal|public|unknown|
|mesd|emotion|Mexican|public|unknown|
|enterface|emotion|Multilingual|public|unknown|
|turev|emotion|Turkish|public|unknown|
|nEMO|emotion,VAD|Polish|public|unknown|
|eaed|emotion|Arabic|public|unknown|
|ased|emotion|Greek|public|unknown|
|cafe|emotion|Childrenspeech, CanadianFrench|public|unknown|
|banglaser|emotion|Bengali|public|unknown|
|mlendsnd|emotion|English|public|unknown|
|gerparas|valence,arousal,dominance|German|restricted|unknown|
|androids|depression|English|public|unknown|
|kbes|emotion|Bengali|public|unknown|
|tess|emotion|British English (Toronto)|public|unknown|
|jtes|emotion|Japanese|private|unknown|
|meld|emotion|English, From Friends TV|public|unknown|
|urdu|emotion|Urdu|public|unknown|
|polish|emotion|Polish|public|unknown|
|cmu-mosei|sentiment,emotion|English, original link dead|public|unknown|
|svd|pahtological speech|German speech data for detecting various pathological voices|public|unknown|
|msp-improv|emotion,VAD,naturalness|English|restricted|unknown|
|shemo|emotion|Persian|public|unknown|
|esd|emotion|English,Chinese|public|unknown|


This recipe contains information about 56 datasets.
## Performance
  
![Nkululeko performance](../meta/images/nkululeko_ser_20240719.png)
