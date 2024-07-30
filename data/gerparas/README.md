# Nkululeko preprocessing for GerPaRas dataset (restricted)

A database of segmented audio files from the German Bundestag. The database contains data from 9 German politicians. After a manual segmentation the data consists of 1198 segments spoken by the nine politicians. The age span was from 40 to 77 years, with 6 men and 3 women (merkel, weidel, nahles).

Test speakers: gauland (102 samples) and weidel (122 samples).

Original audio source: [Deutscher Bundestag](https://www.bundestag.de/) in the year 2020.

All segments were then random spliced as described in the paper. The dataset, which is restricted, can be obtain from [1]. See [2] for details. 

```bash
python3 process_database.py
cd ../..
python3 -m nkululeko.nkululeko --config data/gerparas/exp.ini
# output sample

```

Reference:  
[1] https://zenodo.org/record/7224678  
[2] "Masking speech contents by random splicing: \\Is emotional expression preserved?" Felix Burkhardt, Anna Derington, Matthias Kahlau, Klaus Scherer, Florian Eyben, Bjorn Schuller
