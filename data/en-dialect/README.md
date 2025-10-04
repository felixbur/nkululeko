# Nkululeko pre-processing for Crowdsourced high-quality UK and Ireland English Dialect speech data set. [2]  

This paper presents a dataset of transcribed high-quality audio of English sentences recorded by volunteers speaking with different accents of the British Isles. The dataset is intended for linguistic analysis as well as use for speech technologies. 
Download all links from [1], place it in this directory or somewhere else and




### Make a directory with name 02_process(01_archives is originally for archive just in case)

```bash
mkdir 02_process
```

### Download all links

```bash
wget https://openslr.trmal.net/resources/83/about.html
```

```bash
wget https://openslr.trmal.net/resources/83/LICENSE
```

```bash
wget https://openslr.trmal.net/resources/83/line_index_all.csv
```

```bash
wget https://openslr.trmal.net/resources/83/dialect_info.txt
```

```bash
wget https://openslr.trmal.net/resources/83/irish_english_male.zip
```

```bash
wget https://openslr.trmal.net/resources/83/midlands_english_female.zip
```

```bash
wget https://openslr.trmal.net/resources/83/midlands_english_male.zip
```

```bash
wget https://openslr.trmal.net/resources/83/northern_english_female.zip
```

```bash
wget https://openslr.trmal.net/resources/83/northern_english_male.zip
```

```bash
wget https://openslr.trmal.net/resources/83/scottish_english_female.zip
```

```bash
wget https://openslr.trmal.net/resources/83/scottish_english_male.zip
```

```bash
wget https://openslr.trmal.net/resources/83/southern_english_female.zip
```

```bash
wget https://openslr.trmal.net/resources/83/southern_english_male.zip
```

```bash
wget https://openslr.trmal.net/resources/83/welsh_english_female.zip
```

```bash
wget https://openslr.trmal.net/resources/83/welsh_english_male.zip
```

### Unzipt all zip dataset to 02_process

```bash
unzip '*.zip' -d 02_process
```


### Process the data
```bash
python3 process_database.py
```

### Go back to main directory
```bash
cd ../..
```

### Run experiment
```bash
python3 -m nkululeko.resample --config data/en-dialect/en-dialect_exp.ini
python3 -m nkululeko.nkululeko --config data/en-dialect/en-dialect_exp.ini
```

Reference:  
[1] <https://openslr.org/83/>  
[2] https://storage.googleapis.com/gweb-research2023-media/pubtools/5579.pdf 
