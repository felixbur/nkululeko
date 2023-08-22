# Nkululeko dataset processing for MSP-Podcast dataset

MSP-Podcast can be obtained from [here]https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html).
We used version 1.8.0 of the dataset.

```bash
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/msp-podcast/exp_podcast_audmodel_xgb.ini
```