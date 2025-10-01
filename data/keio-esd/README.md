# Nkululeko pre-processing for KEIO-ESD dataset

```bash
# Process the raw audio files to generate CSV metadata files
python process_database.py # default to use synthesized data
cd ../..
python -m nkululeko.nkululeko data/keio-esd/exp.ini
``` 


# Reference:  
[1] https://research.nii.ac.jp/src/en/Keio-ESD.html