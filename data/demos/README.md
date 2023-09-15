# Nkululeko pre-processing for DEMoS dataset (restrcicted)

DEMoS (Database of Elicited Mood in Speech), is a corpus of induced emotional speech in Italian. DEMoS encompasses 9,365 emotional and 332 neutral samples produced by 68 native speakers (23 females, 45 males) in seven emotional states: the ‘big six’ anger, sadness, happiness, fear, surprise, disgust, and the secondary emotion guilt. To get more realistic productions, instead of acted speech, DEMoS contains emotional speech elicited by combinations of Mood Induction Procedures (MIP). Three elicitation methods are presented, made up by the combination of at least three MIPs, and considering six different MIPs in total. To select samples ‘typical’ of each emotion, evaluation strategies based on self- and external assessment were applied. The selected part of the corpus encompasses 1,564 prototypical samples produced by 59 speakers (21 females, 38 male). DEMoS has been published in the Journal Language, Resousrces, and Evalaution

Download link: [1]

```bash
$ python3 process_database.py
$ python3 -m nkululeko.resample --config data/demos/exp.ini
$ python3 -m nkululeko.nkululeko --config data/demos/exp.ini
```


References:  
[1] https://zenodo.org/record/2544829
[2] Emilia Parada-Cabaleiro, Giovanni Costantini, Anton Batliner, Maximilian Schmitt, and Björn Schuller (2019), DEMoS: An Italian emotional speech corpus. Elicitation methods, machine learning, and perception, Language, Resources, and Evaluation, Feb 2019. https://rdcu.be/bn7oI