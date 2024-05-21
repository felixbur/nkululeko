# Pre-processing nEMO dataset for Nkululeko

This folder contains the pre-processing script of nEMO dataset [1, 2]. Not only (Polish) emotion labels were given, but also text, speaker id, gender and age. Hence, this rich dataset can be used for multitask learning. The process_database.py script will split the entire dataset into train, dev, and test with numbers 3136, 672, and 673 respectively with speaker-independent criteria.

```bash
git clone https://github.com/amu-cai/nEMO
python process_database.py
cd ../.. 
python3 -m nkululeko.resample --config data/nemo/exp.ini
python3 -m nkululeko.nkululeko --config data/nemo/exp.ini
```

Reference:  
[1] <https://github.com/amu-cai/nEMO>  
[2] Christop, I. (2024). nEMO: Dataset of Emotional Speech in Polish. <http://arxiv.org/abs/2404.06292>
