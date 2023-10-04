# Nkululeko pre-processing for ASED (V1) database (public)

In this paper we present the Amharic Speech Emotion Dataset (ASED), which covers four dialects
(Gojjam, Wollo, Shewa and Gonder) and five different emotions (neutral, fearful, happy, sad and
angry). We believe it is the first Speech Emotion Recognition (SER) dataset for the Amharic language.
65 volunteer participants, all native speakers of Amharic, recorded 2,474 sound samples, two to
four seconds in length. Eight judges (two for each dialect) assigned emotions to the samples with
high agreement level (Fleiss kappa = 0.8). The resulting dataset is freely available for download.
Next, we developed a four-layer variant of the well-known VGG model which we call VGGb. Three
experiments were then carried out using VGGb for SER, using ASED. First, we investigated whether
Mel-spectrogram features or Mel-frequency Cepstral coefficient (MFCC) features work best for
Amharic. This was done by training two VGGb SER models on ASED, one using Mel-spectrograms
and the other using MFCC. Four forms of training were tried, standard cross-validation, and three
variants based on sentences, dialects and speaker groups. Thus, a sentence used for training would
not be used for testing, and the same for a dialect and speaker group. The conclusion was that MFCC
features are superior under all four training schemes. MFCC was therefore adopted for Experiment
2, where VGGb and three other existing models were compared on ASED: RESNet50, Alex-Net
and LSTM. VGGb was found to have very good accuracy (90.73%) as well as the fastest training
time. In Experiment 3, the performance of VGGb was compared when trained on two existing SER
datasets, RAVDESS (English) and EMO-DB (German) as well as on ASED (Amharic). Results are
comparable across these languages, with ASED being the highest. This suggests that VGGb can be 
successfully applied to other languages. We hope that ASED will encourage researchers to explore
the Amharic language and to experiment with other models for Amharic SER.

Download link [2].


```bash
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/ase/exp.ini
```

Reference:  
[1] Retta, Ephrem Afele, Eiad Almekhlafi, Richard Sutcliffe, Mustafa Mhamed, Haider Ali, and Jun Feng. "A new Amharic speech emotion dataset and classification benchmark." ACM Transactions on Asian and Low-Resource Language Information Processing 22, no. 1 (2023): 1-22.  
[2] https://github.com/Ethio2021/ASED_V1