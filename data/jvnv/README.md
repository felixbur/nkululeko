# Nkululeko pre-processing for JNVV database [1]  

```bash
wget https://ss-takashi.sakura.ne.jp/corpus/jvnv/jvnv_ver1.zip
unzip jvnv_ver1.zip
# process the database, need to add nkululeko to use find_files function
python3 process_database.py
cd ../..
# resample to 16k
python3 -m nkululeko.resample --config data/jvnv/exp.ini
# make experiment
python3 -m nkululeko.nkululeko --config data/jvnv/exp.ini
# sample results (exp.ini):
# DEBUG reporter: epoch: 0, UAR: .918, (+-.882/.948), ACC: .919
```

References:  
[1] Xin, D., Jiang, J., Takamichi, S., Saito, Y., Aizawa, A., & Saruwatari, H. (2023). JVNV: A Corpus of Japanese Emotional Speech with Verbal Content and Nonverbal Expressions. 1–12. <http://arxiv.org/abs/2310.06072>
