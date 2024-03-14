# nkululeko.demo

Module to generate predictions from unlabeled data using a pre-trained model.

Usage:  
```bash
python -m nkululeko.demo --config my_config.ini --list my_testsamples.csv --outfile my_results.csv
```

Example of a configuration file:
```ini
[EXP]
save = True
[MODEL]
save = True
```