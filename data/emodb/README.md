EmoDB can be downloaded from [Zenodo](https://zenodo.org/record/7447302).

Simply download and unzip, it is already in audformat.

```bash
# Download using wget
wget https://zenodo.org/record/7447302/files/emodb.zip
# Unzip
unzip emodb.zip
# change to Nkululeko parent directory
cd ../..
# run the nkululeko experiment
python -m nkululeko.nkululeko --config tests/exp_emodb_os_xgb.ini
```

Then, check the results in the `results` directory.