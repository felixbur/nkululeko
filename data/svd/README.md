# Nkululeko prerocessing for Saarbruecken Voice Database (SVD) dataset

This is Nkululeko pre-processing recipe for the SVD dataset.

# Filenaming convention (A to H correspond to columns in metadata list files)

A: ID, identification number of recording  
B: T, type of recording (n:normal, p:pathological)  
C: D, date of recording   
D: S, identification number of the speaker  
E: G, gender of the speaker (w:woman, m:man)  
F: A, age of the speaker at the time of recording  
G: Pathologies  
H: Remark with regard to diagnosis  

```bash
$ python3 process_database.py
# Download and extract SVD dataset, see exp.ini for configuration
$ cd ../..
$ python3 -m nkululeko.nkululeko --config data/svd/exp.ini
```
[1] https://stimmdb.coli.uni-saarland.de/index.php4