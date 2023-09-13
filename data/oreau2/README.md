# Nkululeko pre-processing for OREAU2 dataset

This workflow assumes the second version of Oreau [2]. See below for the reason.  

This document presents the French emotional speech database - Oréau, recorded in a quiet environment. The database is designed for general study of emotional speech and analysis of emotion characteristics for speech synthesis purposes. It contains 79 utterances which could be used in everyday life in the classroom. Between 10 and 13 utterances were written for each of the 7 emotions in French language by 32  non-professional  speakers.

2 versions are available, the first one contains 502 sentences. A perception test was performed to evaluate the recognition of emotions and their naturalness. 90% of utterances (434 utterances) were correctly identified and retained after the test and various analyses, which constitutes the second version of database. The versions are available on Zenodo [1].

```bash
$ python3 process_database.py
$ python3 -m nkululeko.resample --config data/oreau2/exp.ini
$ python3 -m nkululeko.nkululeko --config data/oreau2/exp.ini
```


References:  
[1] https://zenodo.org/record/4405783