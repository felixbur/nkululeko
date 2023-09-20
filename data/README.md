# Data

This is the default top directory for data import for Nkululeko.

Each database should be in its own subfolder (you can also use `ln -sf`` to soft link original database path to these subfolders) and contain a README how to import the data to Nkululeko CSV or audformat.

## Accesibility

The column `acess` in the table below indicates the accessability of the database. The following values are used:
- `public`: the database is publicly available in the internet and can be downloaded directly without any restrictions.
- `restricted`: the database is publicly available in the internet but requires registration or other restrictions to download.
- `private`: the database is not publicly available in the internet and requires to contact privately to the owner of the dataset.

## Databases

| # |name | target | acess | descr. |
| --|---- | ------ | ----- | -------|
| 1 | aesdd | emotion | public | amharic language |
| 2 | androids | depression | public | |
| 3 | ased | emotion | public |
| 4 | asvp-esd | emotion | public |
| 5 | baved | emotion | public |
| 6 | cafe | emotion | public |
| 7 | cmu-mosei | sentiment, emotion| public |
| 8 | crema-d | emotion | public |
| 9 | demos | emotion | restricted | Italian |
|10 | ekorpus | emotion | public | Estonian |
|11 | emns | emotion | public | Estonian |
|12 | emodb | emotion | public | German   UAR=0.479 |
|13 | emofilm | emotion | restricted | English, Spanish, Italian |
|14 | emorynlp | emotion | public |  |   
|15 | emov-db | emotion | public |
|16 | emovo | emotion | restricted | Italian |
|17 | emozionalmente | emotion | public | Italian |
|18 | enterface | emotion | public | |
|19 | esd | emotion| public | English, Chinese|
|20 | iemocap | emotion, VAD | restriced | English |
|21 | jl | emotion | public |
|22 | jtes | emotion | private | Japanese |
|23 | laughter-types | laughter | public | |
|24 | meld | emotion | public | |
|25 | mesd | emotion | public | |
|26 | mess | emotion | public | |
|27 | mlendsnd | emotion | public |
|28 | msp-improv | emotion, VAD, naturalness | restricted |
|29 | msp-podcast | emotion, VAD | restricted |
|30 | oreau2 | emotion | public | |
|31 | portuguese | emotion | public |  Portuguese |
|32 | ravdess | emotion | public | English |
|33 | savee | emotion | restricted | |
|34 | shemo | emotion | public | |
|35 | subesco | emotion | public | Bangla |
|36 | syntact | emotion | public | Synthesized German speech |
|37 | tess | emotion | public | |
|38 | thorsten-emotional | emotion | public |
|29 | urdu | emotion | public | |
|40 | vivae | emotion | public | |