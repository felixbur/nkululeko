# Nkululeko pre-processing for Portuguese dataset (Public)

A set of semantically neutral sentences and derived pseudosentences was produced by two native European
Portuguese speakers varying emotional prosody in order to portray anger, disgust, fear, happiness, sadness, surprise, and neutrality. Accuracy rates and reaction times in a forced-choice identification of these emotions as well as intensity judgments were collected from 80 participants, and a database was constructed with the utter- ances reaching satisfactory accuracy (190 sentences and 178 pseudosentences). High accuracy (mean correct of 75% for sentences and 71% for pseudosentences), rapid recognition, and high-intensity judgments were obtained for all the portrayed emotional qualities. Sentences and pseudosentences elicited similar accuracy and intensity rates, but participants responded to pseudosentences faster than they did to sentences. This database is a useful tool for research on emotional prosody, including cross-language studies and studies involving Portuguese- speaking participants, and it may be useful for clinical purposes in the assessment of brain-damaged patients. The database is available for download from http://brm.psychonomic-journals.org/content/supplemental.

Download link: https://static-content.springer.com/esm/art%3A10.3758%2FBRM.42.1.74/MediaObjects/Castro-BRM-2010%20Sents.zip

```bash
$ unzip unzip Castro-BRM-2010\ Sents.zip
$ mv Castro-BRM-2010\ Sents PORTUGUESE
$ unzip Castro-BRM-2010\ Pseudosents.zip -d PORTUGUESE/
$ cd PORTUGUESE
$ mv ' Castro_2010_AppxB_Pseudosents.txt'  Castro_2010_AppxB_Pseudosents.txt
$ mv ' Castro_2010_AppxB_Sents.txt'  Castro_2010_AppxB_Sents.txt
$ python3 -m nkululeko.nkululeko --config data/portuguese/exp.ini
```

Reference: 
[1] Castro, S. L., & Lima, C. F. (2010). Recognizing emotions in spoken language: A validated set of Portuguese sentences and pseudosentences for research on emotional prosody. Behavior Research Methods, 42(1), 74–81. https://doi.org/10.3758/BRM.42.1.74