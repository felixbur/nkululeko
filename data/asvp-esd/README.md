# Nkululeko preprocessing for ASVP-ESD dataset (public)

The Audio, Speech, and Vision Processing Lab Emotional  Sound database (ASVP-ESD)

Dejoli Tientcheu Touko Landry; Qianhua He; Wei Xie

Citing the ASVP-ESD:
The ASVP-ESD emotional sound database is released by the Audio, Speech, and Vision Processing Lab(http://www.speech-led.com/main.htm, from the South China University of Technology), so please cite the ASVP-ESD: A dataset and its benchmark for emotion recognition using both speech and non-speech utterances [2] (papers which is a study conducted using the first batch of the collected database) if it is used in your work in any form. Personal works, such as machine learning projects or posts, should provide a URL to Zenodo page[1], through a reference.

Contact Information:
If you would like further information about the ASVP-ESD, when; facing any issues downloading files, please contact the authors at: 201722800077@mail.scut.edu.cn, 1197581424@qq.com

```
$ wget https://zenodo.org/record/7132783/files/ASVP-ESD-Update.zip
$ unzip ASVP-ESD-Update.zip
$ python process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/asvp-esd/exp.ini
```
References:  
[1] https://zenodo.org/record/7132783  
[2] https://www.kaggle.com/datasets/dejolilandry/asvpesdspeech-nonspeech-emotional-utterances
