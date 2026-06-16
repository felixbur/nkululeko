# StressDat

This is about the [Slovak stress dataset](https://www.sav.sk/journals/uploads/11080936579-589-stressdat-database-of-speech-under-stress-in-slovak.pdf), try to get a version by contacting the authors.

Then unzip to a folder named *repo*

Then do

* then do ```uv run 01_generate_tables.py``` to create data table files
* ```uv run 02_generate_db.py``` to create the audformat database

To check the database, run an nkululeko experiment with

```python -m nkululeko.nkululeko --config exp.ini```
