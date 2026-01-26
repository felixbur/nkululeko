# Nkululeko pre-processing for Fake or Real(FoR) - 2 seconds  

## Dataset description  

The Fake-or-Real (FoR) dataset [1,2] is a collection of more than 195,000 utterances from real humans and computer generated speech. The dataset can be used to train classifiers to detect synthetic speech.

The dataset aggregates data from the latest TTS solutions (such as Deep Voice 3 and Google Wavenet TTS) as well as a variety of real human speech, including the Arctic Dataset (http://festvox.org/cmu_arctic/), LJSpeech Dataset (https://keithito.com/LJ-Speech-Dataset/), VoxForge Dataset (http://www.voxforge.org) and our own speech recordings.

The dataset is published in four versions: for-original, for-norm, for-2sec and for-rerec. The third one, named for-2sec is based on the second one, but with the files truncated at 2 seconds. Since this the smallest version of the dataset, it is suitable for quick experiments. It also gains the most performance in AUDDT paper compared to other three variants [3]. 

## Pre-processing command

Download link: [https://bil.eecs.yorku.ca/share/for-2sec.tar.gz](https://bil.eecs.yorku.ca/share/for-2sec.tar.gz)

```bash
# if downloading from original dataset
wget https://bil.eecs.yorku.ca/share/for-2sec.tar.gz
tar -xvzf for-2sec.tar.gz

# Generate CSV files
python3 process_database.py

# Run experiment with UAR metric (default)
cd ../..
python3 -m nkululeko.nkululeko --config data/for-2sec/exp.ini

# Or run with EER (Equal Error Rate) metric for deepfake detection
python3 -m nkululeko.nkululeko --config data/for-2sec/exp_eer.ini
```

## EER Metric Support

This dataset is ideal for testing the **Equal Error Rate (EER)** metric, which is commonly used in biometric systems and deepfake detection. The `exp_eer.ini` configuration demonstrates how to use EER as the primary metric while still reporting UAR.

See [docs/EER_IMPLEMENTATION.md](../../docs/EER_IMPLEMENTATION.md) for details about the EER implementation.

Reference:  
[1] Reimao, Ricardo, and Vassilios Tzerpos. "For: A dataset for synthetic speech detection." In 2019 International Conference on Speech Technology and Human-Computer Dialogue (SpeD), pp. 1-10. IEEE, 2019.  
[2] https://bil.eecs.yorku.ca/datasets/  
[3] Zhu, Yi, Heitor R. Guimarães, Arthur Pimentel, and Tiago Falk. "AUDDT: Audio Unified Deepfake Detection Benchmark Toolkit." arXiv preprint arXiv:2509.21597 (2025).  