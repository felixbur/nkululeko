# Nkululeko preprocsssing for CaFE dataset (public)

The Canadian French Emotional (CaFE) speech dataset contains six different sentences, pronounced by six male and six female actors, in six basic emotions plus one neutral emotion. The six basic emotions are acted in two different intensities: mild ("Faible") and strong ("Fort").

We used 48kHz (sampling rate) version of the dataset and resampled them to 16kHz.

Download link: [1], also see wget command below.

```bash 
$ https://zenodo.org/record/1478765/files/CaFE_48k.zip
$ unzip CaFE_48k.zip -d CaFE
$ python3 process_database.py
$ python3 -m nkululeko.resample data/cafe/exp.ini
$ python3 -m nkululeko.nkululeko data/cafe/exp.ini
```


References:   
[1] https://zenodo.org/record/1478765  
[2] P. Gournay, O. Lahaie, and R. Lefebvre, “A Canadian French emotional speech dataset,” in Proc. ACM Multimedia Systems, 2018, pp. 399–402.