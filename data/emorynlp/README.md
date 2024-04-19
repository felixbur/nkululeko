# Nkululeko pre-processing for EmoryNLP dataset (public)

Download link: https://drive.google.com/u/0/uc?id=1UQduKw8QTqGf3RafxrTDfI1NyInYK3fr&export=download

Please read MELD repository [1] for more information.

```bash
$ unzip emorynlp_video_splits.zip
$ mkdir EMORYNLP
$ unzip emorynlp_train_splits.zip -d EMORYNLP
$ unzip emorynlp_test_splits.zip -d EMORYNLP
$ unzip emorynlp_dev_splits.zip -d EMORYNLP
# copy from https://github.com/declare-lab/MELD/blob/master/data/emorynlp/
# into EMORYNLP folder
$ python3 process_database.py
```

References:    
[1] https://github.com/declare-lab/MELD  
[2] S. Zahiri and J. D. Choi. Emotion Detection on TV Show Transcripts with Sequence-based Convolutional Neural Networks. In The AAAI Workshop on Affective Content Analysis, AFFCON'18, 2018.  
[3] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversation. ACL 2019.
