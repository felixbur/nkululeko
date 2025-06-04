# Nkululeko pre-processing for URDU dataset (Public)

Cross-lingual speech emotion recognition (SER) is a
crucial task for many real-world applications. The performance of SER systems is often degraded by the differences in the distributions of training and test data. These differences become more apparent when training and test data belong to different languages, which cause a significant performance gap between the validation and test scores. It is imperative to build more robust models that can fit in practical applications of SER systems. Therefore, in this paper, we propose a Generative Adversarial Network (GAN)-based model for multilingual SER. Our choice of using GAN is motivated by their great success in learning the underlying data distribution. The proposed model is designed in such a way that can learn language invariant representations without requiring target-language data labels. We evaluate our proposed model on four different language emotional datasets, including an Urdu-language dataset to also incorporate alternative languages for which labelled data is difficult to find and which have not been studied much by the mainstream community. Our results show that our proposed model can significantly improve the baseline cross-lingual SER performance for all the considered datasets including the non- mainstream Urdu language data without requiring any labels.

Download ZIP from [1]; it is also available at [3].


```bash
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/urdu/exp.ini
```


References:  
[1] https://github.com/siddiquelatif/URDU-Dataset/  
[2] Latif, S., Qadir, J., & Bilal, M. (2019). Unsupervised Adversarial Domain Adaptation for Cross-Lingual Speech Emotion Recognition. 2019 8th International Conference on Affective Computing and Intelligent Interaction, ACII 2019. https://doi.org/10.1109/ACII.2019.8925513  
[3] https://github.com/bagustris/multilingual_speech_emotion_dataset  