---
title: 'Nkululeko 1.0: A Python package to predict speaker characteristics with a high-level interface'
tags:
  - Python
  - speech
  - machine learning
  - data exploration
  - speaker characteristics
authors:
  - name: Felix Burkhardt
    orcid: 0000-0002-2689-0545
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Bagus Tris Atmaja
    orcid: 0000-0003-1560-2824
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 3
affiliations:
 - name: audEERING GmbH, Germany
   index: 1
 - name: TU Berlin, Germany
   index: 2
 - name: Nara Institute of Science and Technology (NAIST), Japan
   index: 3
date: 15 April 2025  # Date of JOSS Review, submitted 2024-12-24
bibliography: paper.bib

---

# Summary

`Nkululeko` [@nkululeko:2022] is a Python toolkit for audio-based machine learning that uses a command-line interface and configuration files, eliminating the need for users to write code. Built on sklearn [@scikit-learn:2011] and PyTorch [@torch:2020], it enables training and evaluation of speech databases with state-of-the-art machine learning approaches and acoustic features. Key capabilities include model demonstration, database storage with predicted labels, and bias detection through correlation analysis of target labels (e.g., depression) with speaker characteristics (age, gender) or signal quality metrics.

# Design Choices 

Nkululeko targets **novice users** interested in speaker characteristics detection (emotion, age, gender) without programming expertise, focusing on **education** and **research**. Core design principles include: (1) exploring combinations of acoustic features, models, and preprocessing for optimal performance; (2) database analysis with visualizations; (3) inference on audio files or streams. Users run experiments via a single command: `nkululeko.MODULE_NAME --config CONFIG_FILE.ini`.

# How Does It Work?

Nkululeko is a Python command-line tool that uses INI configuration files to specify experiments. Data is imported via CSV format (file path, speaker ID, gender, task labels) or audformat. The functionality is encapsulated by software modules that are called on the command line. Key modules include:

* **nkululeko**: machine learning experiments combining features and learners (e.g., opensmile with SVM);
* **explore**: data exploration and analysis with visualizations;
* **predict**: predict features like speaker diarization, signal distortion ratio, mean opinion score, age/gender with deep learning models;
* **segment**: segment database based on VAD (voice activity detection);
* **ensemble**: combine several models to improve performance;
* **demo**: demonstrate the current best model on command line or files;
* **augment**: augment training data for bias reduction;
* **optim**: search model's best hyperparamaters;
* **flags**: run several experiments at once.

Configuration files contain sections: DATA (database location, target labels), FEATS (acoustic features: opensmile [@opensmile:2010], wav2vec 2.0 [@wav2vec:2020]), MODEL (classifiers/regressors), and PLOT (visualization). The overall workflow is shown in \autoref{fig:nkulu_flow}. Results include images, text reports, and auto-generated LaTeX/PDF documentation.

![Nkululeko's workflow: from raw dataset to experiment results \label{fig:nkulu_flow}](./assets/nkulu_flow-crop.pdf)

# Statement of Need

Open-source tools accelerate science through security, customizability, and transparency. While several open-source tools exist for audio analysis—librosa [@McFee:2015], TorchAudio [@Yang:2021], pyAudioAnalysis [@Giannakopoulos:2015], ESPNET [@Watanabe:2018], and SpeechBrain [@speechbrain:2021]—none specialize in speech analysis with high-level interfaces for novices. Nkululeko fills this gap with key principles: 

1. minimal programming skills (CSV data preparation and command-line execution); 

1. standardized data formats (CSV and AUDFORMAT); 

1. replicability through shareable configuration files; 

1. high-level INI-file interface requiring no Python coding; 

1. transparency via comprehensive debug output and automated reporting. 

Nkululeko interfaces with Spotlight [@spotlight:2023] for enhanced metadata visualization, combining complementary functionalities.

# Usage in Existing Research

Nkululeko has been used in several research projects since 2022 [@nkululeko:2022]:

- [@burkhardt:2022-syntact] evaluated synthesized emotional speech databases;

- [@Burkhardt:2024] demonstrated bias detection in UACorpus and Androids datasets; 

- [@Atmaja:2024a] showcased ensemble learning with uncertainty estimation;

- [@Atmaja:2025] evaluated handcrafted acoustic features and self-supervised learning for pathological voice detection with early/late fusion strategies; 

- [@Atmaja:2025b] extended ensemble evaluations with performance weighting across five tasks and ten datasets.

# Acknowledgements

We acknowledge support from: European SHIFT project (Grant 101060660); European EASIER project (Grant 101016982); Project JPNP20006 (NEDO, Japan); Project 24K02967 (JSPS). We thank audEERING GmbH for partial funding.

# References
