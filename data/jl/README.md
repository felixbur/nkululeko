# Nkululeko pre-processing for JL corpus

For further understanding the wide array of emotions embedded in human speech, we are introducing an emotional speech corpus. In contrast to the existing speech corpora, this corpus was constructed by maintaining an equal distribution of 4 long vowels in New Zealand English. This balance is to facilitate emotion related formant and glottal source feature comparison studies. Also, the corpus has 5 secondary emotions along with 5 primary emotions. Secondary emotions are important in Human-Robot Interaction (HRI), where the aim is to model natural conversations among humans and robots. But there are very few existing speech resources to study these emotions,and this work adds a speech corpus containing some secondary emotions.

The dataset can be downloade from [2]; it is also included in [3].

```
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/jl/exp.ini
```

References:  
[1] Jesin James, Li Tian, Catherine Watson, "An Open Source Emotional Speech Corpus for Human Robot Interaction Applications", in Proc. Interspeech, 2018.  
[2] https://www.kaggle.com/datasets/tli725/jl-corpus  
[3] https://github.com/bagustris/multilingual_speech_emotion_recognition_datasets