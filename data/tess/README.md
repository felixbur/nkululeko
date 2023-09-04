# Nkululeko dataset preprocessing for TESS dataset

TESS (Toronto Emotional Speech Set) is a dataset of emotional speech. These stimuli were modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966). A set of 200 target words was spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 stimuli in total.
Two actresses were recruited from the Toronto area. Both actresses speak English as their first language, are university educated, and have musical training. Audiometric testing indicated that both actresses have thresholds within the normal range.

Authors: Kate Dupuis, M. Kathleen Pichora-Fuller

The dataset is available for download at [1] and [2].

```bash
# Download the dataset (dir name "TESS")
python3 process_database.py
cd ../..
python3 -m nkululeko.nkululeko --config ./data/tess/exp.ini
```

References:  
[1] https://tspace.library.utoronto.ca/handle/1807/24487  
[2] https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess