This folder is to import the 
Androids-corpus (androids)
containing data from 112 people, 64 of them diagnozed with depression and 52 as a control group.
Described in the upcoming paper [1].

I used the version downloadable from [Dropbox, mentioned in this github page](https://github.com/androidscorpus/data)

Download and unzip the file Androids-corpus.zip to the current folder.


```bash
# Copy from download folder to here
$ cp ~/Downloads/Androids-corpus.zip .
# Unzip
$ unzip Androids-corpus.zip
# Preprocess the data
$ python process_database.py
# change to root folder
$ cd ../..
# run nkululeko experiments
$ python -m nkululeko.explore --config tests/exp_androids_explore.ini
```

Reference:  
[1] Tao, F., Esposito, A., Vinciarelli, A. (2023) The Androids Corpus: A New Publicly Available Benchmark for Speech Based Depression Detection. Proc. INTERSPEECH 2023, 4149-4153, doi: 10.21437/Interspeech.2023-894  