# Nkululeko pre-processing for EmoChildRu Dataset

```bash
# grad the dataset from github
git clone https://github.com/hydra-colab/Emotion-Recognition-of-Younger-School-Age-Children

# rename WAV with wav
find Emotion-Recognition-of-Younger-School-Age-Children -name '*.WAV' -exec bash -c 'f="{}"; mv -- "$f" "${f%.WAV}.wav"' \;
# check number of file, should be 2505
find Emotion-Recognition-of-Younger-School-Age-Children/ -name '*.wav' | wc -l
# process database
python3 process_database.py
# resample and run
cd ../..
python3 -m nkululeko.resample --config data/erysac/exp.ini
python3 -m nkululeko.nkululeko --config data/erysac/exp.ini
```


Reference:  
[1] E. Lyakso et al., “EmoChildRu: Emotional child Russian speech corpus,” Lect. Notes Comput. Sci. (including Subser. Lect. Notes Artif. Intell. Lect. Notes Bioinformatics), vol. 9319, pp. 144–152, 2015, doi: 10.1007/978-3-319-23132-7_18.