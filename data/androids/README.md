Androids can be downloaded from [Dropbox](https://zenodo.org/record/7447302).

Simply download and unzip, it is already in audformat.

```bash
# Download using wget
wget wget https://www.dropbox.com/s/2rurxlh70ihfki4/Androids-Corpus.zip?dl=0
# rename
mv Androids-Corpus.zip\?dl\=0 Androids-Corpus.zip
# Unzip
unzip Androids-Corpus.zip
# delete zip file for space
rm Androids-Corpus.zip
# change to Nkululeko parent directory
cd ../..
# convert to mono 16 kHz sampling rate
python -m nkululeko.resample --config data/androids/exp.ini
# explore the data
python -m nkululeko.explore --config data/androids/exp.ini
# run the nkululeko experiment
python -m nkululeko.nkululeko --config data/androids/exp.ini
```

Then, check the results in the `results` directory.
