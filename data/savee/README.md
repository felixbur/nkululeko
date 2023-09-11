# Nkululeko pre-processing for SAVEE dataset (restricted)
SAVEE (Surrey Audio-Visual Expressed Emotion) is an emotion recognition dataset. It consists of recordings from 4 male actors in 7 different emotions, 480 British English utterances in total. The sentences were chosen from the standard TIMIT corpus and phonetically-balanced for each emotion. This release contains only the audio stream from the original audio-visual recording.

The data is split so that the training set consists of 2 speakers, and both the validation and test set consists of samples from 1 speaker, respectively.

The dataset folder is provided upon registration. You need to register at http://personal.ee.surrey.ac.uk/Personal/P.Jackson/SAVEE/Register.html in order to get the link to download the dataset.


```bash
$ unzip AudioData.zip
# You can unzip the data in other path and make softlink (ln -sf) to here
$ mv AudioData SAVEE
$ python3 process_database.py
$ cd ../..
$ python3 nkululeko.resample --config data data/savee/exp.ini
$ python3 nkululeko.nkululeko --config data data/savee/exp.ini

```

References:  
[1] Vlasenko, B., Schuller, B., Wendemuth, A., & Rigoll, G. (2007). Combining frame and turn-level information for robust recognition ofemotions within speech. Proc. Interspeech 2007, 4, 2712–2715.