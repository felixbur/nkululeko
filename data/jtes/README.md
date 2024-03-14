# Nkululeko dataset processing for JTES dataset

JTES (Japanese Twitter-based Emotional Speech) corpus is private dataset developed by 
Tohoku university [1]. The dataset is not publicly available, but can be requested from the authors.
The partition here used text-independent evaluation as described in [2].

```bash
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/jtes/exp_jtes_audmodel_svm.ini
```




References:  
[1] E. Takeishi, T. Nose, Y. Chiba, and A. Ito, ‘‘Construction and analysis of phonetically and prosodically balanced emotional speech database,’’ in Proc. Conf. Oriental Chapter Int. Committee Coordination Standard- ization Speech Databases Assessment Techn. (O-COCOSDA), Oct. 2016, pp. 16–21.  
[2] Atmaja, B.T.; Sasou, A. Sentiment Analysis and Emotion Recognition from Speech Using Universal Speech Representations. Sensors 2022, 22, 6369. https:// doi.org/10.3390/s22176369
Academic