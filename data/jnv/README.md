# JNV Corpus  

Pre-processing the JNV Corpus for Nkululeko (CSV format).

```bash
wget https://ss-takashi.sakura.ne.jp/corpus/jnv/jnv_corpus_ver2.zip
unzip jnv_corpus_ver2.zip
python3 process_database.py
cd ../..
# the following will resample and replace JNV forpus to 16k 
python3 -m nkululeko.resample --config data/jnv/exp.ini
python3 -m nkululeko.nkululeko --config data/jnv/exp.ini
```