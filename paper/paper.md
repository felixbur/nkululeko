---
title: 'Nkululeko: A Python package to predict speaker characteristics with a high-level interface.'
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
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: audEERING GmbH, Germany
   index: 1
 - name: TU Berlin, Germany
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 22 October 2024
bibliography: paper.bib

---

# Summary

`Nkululeko` [@nkululeko:2022] is open-source software written in Python and hosted on GitHub.
It is predominantly a framework for audio-based machine learning explorations without the need to write Python code, and is strongly based on machine learning packages like sklearn [@scikit-learn:2011] and pytorch [@torch:2020].
The main features are: training and evaluation of labelled speech databases with state-of-the-art machine learning approach and acoustic feature extractors, a live demonstration interface, and the possibility to store databases with predicted 
labels. Based on this, the framework can be used to check on bias in databases by exploring correlations of target labels, like, e.g. depression or diagnosis, with predicted, or additionally given, labels like age, gender, STOI, SDR, or MOS.


# Statement of need
Open-source tools are believed to be one of the reasons for accelerated science and technology. They are more secure, easy to customise and transparent. There are several open-source tools that exist for acoustic, sound, and audio analysis, such as librosa [@McFee:2015], TorchAudio [@Yang:2021], pyAudioAnalysis [@Giannakopoulos:2015], ESPNET [@Watanabe:2018], and SpeechBrain [@speechbrain:2021]. However, none of them are specialised in speech analysis with high-level interfaces for novices in the speech processing area. 

One exception is Spotlight [@spotlight:2023], an open-source tool that visualises metadata distributions in audio data. An existing interface between `nkulu` and Spotlight can be used to combine the visualisations of Spotlight with the functionalities of Nkululeko.


# Acknowledgements

We acknowledge support from .

# References
