# Nkululeko pre-processing for EMNS dataset (public)


Emotive Narrative Storytelling (EMNS) corpus introduces a dataset consisting of a single speaker, British English speech with high-quality labelled utterances tailored to drive interactive experiences with dynamic and expressive language. Each audio-text pairs are reviewed for artefacts and quality. Furthermore, we extract critical features using natural language descriptions, including word emphasis, level of expressiveness and emotion. 


Download link: http://www.openslr.org/136/.

```bash
$ wget https://www.openslr.org/resources/136/raw_webm.tar.xz
$ wget https://www.openslr.org/resources/136/metadata.csv
$ tar -xf raw_webm.tar.xz
$ mv raw_webm EMNS
$ mv metadata.csv EMNS
$ python3 process_database.py
$ python3 -m nkululeko.resample --config data/emns/configs/exp_resample.ini
$ python3 -m nkululeko.nkululeko --config data/emns/configs/exp.ini
```


Reference:  
[1] Noriy, K. A., Yang, X., & Zhang, J. J. (2023). EMNS /Imz/ Corpus: An emotive single-speaker dataset for narrative storytelling in games, television and graphic novels. https://github.com/knoriy/EMNS-DCT