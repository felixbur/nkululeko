EmoDB can be downloaded from [Zenodo](https://zenodo.org/records/16273947).

Simply download and unzip, it is already in audformat.

```bash
# Download using wget
wget https://zenodo.org/record/16273947/files/emodb_2.0.zip
# Unzip
unzip emodb_2.0.zip
# rename to simply emodb
mv emodb_2.0 emodb
# change to Nkululeko parent directory
cd ../..
# run the nkululeko experiment
python -m nkululeko.nkululeko --config tests/exp_emodb_os_xgb.ini
```

Then, check the results in the `results` directory.
