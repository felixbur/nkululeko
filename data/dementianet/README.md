# Nkululeko pre-processing for Dementianet  

DementiaNet [1] is a longitudinal spontaneous speech (audio) dataset for dementia screening. These are audio samples scrapped from youtube for public figures and celebrities:  

- who had confirmed dementia diagnoses for Dementia data  
- who lived on to beyond 90 years without any noticeable decline in cognitive health for No-dementia data.  

The sample dataset contains a hundred individuals with a confirmed dementia diagnosis. Spontaneous speech samples (audio) range from time after the confirmed diagnosis to ten years before the symptoms appear. And a hundred individuals over the age of eighty with no cognitive decline (NC) and active in their field of work. Spontaneous speech samples for the NC group fall into three buckets, five years, ten years and fifteen years before death or current age. Early analysis of this dataset shows above 70% accuracy. According to our knowledge, DementiaNet is the largest publicly available longitudinal dataset for dementia prediction/screening.  

```bash
$ mkdir DEMENTIANET
$ unzip dementia-20250604T010747Z-1-001.zip -d DEMENTIANET
$ unzip unzip nodementia-20250604T010819Z-1-002.zip -d DEMENTIANET
$ python3 process_database.py
$ cd ..
$ python3 -m nkululeko.nkululeko --config data/dementianet/exp.ini
```

Reference:  
[1] https://github.com/shreyasgite/dementianet  