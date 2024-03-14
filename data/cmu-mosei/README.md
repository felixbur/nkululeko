# Nkululeko dataset processing for CMU-MOSEI dataset

**NOTE** that raw CMU-MOSEI dataset currently is not available from the original link (http://immortal.multicomp.cs.cmu.edu/raw_datasets/). You can request the dataset from the authors of the paper [CMU-MOSEI: Multimodal Sentiment Analysis Dataset](https://arxiv.org/abs/1606.06259) by Zadeh et al. (2018).

```bash
# process the dataset
$ unzip CMU_MOSEI.zip
$ mv Raw CMU-MOSEI
$ python3 process_database.py
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/cmu-mosei/exp.ini
```