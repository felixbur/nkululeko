# Test module

Module for testing a saved model on known datatast (has labels).  

Usage:  

```bash
python -m nkululeko.test --config myconfg.ini --outfile myresults.csv
```

Example of `INI` file:  

```ini
[DATA]
tests = ['my_testdb']
my_testdb = /mypath/my_testdb
...
```
