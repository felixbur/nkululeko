# Nkululeko pre-processing for MELD dataset

Multimodal EmotionLines Dataset (MELD) has been created by enhancing and extending EmotionLines dataset. MELD contains the same dialogue instances available in EmotionLines, but it also encompasses audio and visual modality along with text. MELD has more than 1400 dialogues and 13000 utterances from Friends TV series. Multiple speakers participated in the dialogues. Each utterance in a dialogue has been labeled by any of these seven emotions -- Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear. MELD also has sentiment (positive, negative and neutral) annotation for each utterance [1].

The dataset is available in [3]. This worfklow assumes the dataset is download from [2].

```bash
$ python3 process_database.py
$ python3 -m nkululeko.nkululeko --config data/meld/exp.ini
```


References:  
[1] S. Poria, D. Hazarika, N. Majumder, G. Naik, R. Mihalcea,
E. Cambria. MELD: A Multimodal Multi-Party Dataset
for Emotion Recognition in Conversation. (2018)  
[2] https://github.com/bagustris/multilingual_speech_emotion_recognition_datasets  
[3] https://github.com/declare-lab/MELD