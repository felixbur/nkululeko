# Nkululeko pre-processing for Crowdsourced high-quality UK and Ireland English Dialect speech data set. [2]  

This paper presents a dataset of transcribed high-quality audio of English sentences recorded by volunteers speaking with different accents of the British Isles. The dataset is intended for linguistic analysis as well as use for speech technologies. 
Download all links from [1], place it in this directory or somewhere else and



```bash
#make a directory with name 02_process(01_archives is originally for archive just in case)
mkdir 02_process

#download all links
wget https://openslr.trmal.net/resources/83/about.html
wget https://openslr.trmal.net/resources/83/LICENSE
wget https://openslr.trmal.net/resources/83/line_index_all.csv
wget https://openslr.trmal.net/resources/83/dialect_info.txt
wget https://openslr.trmal.net/resources/83/irish_english_male.zip
wget https://openslr.trmal.net/resources/83/midlands_english_female.zip
wget https://openslr.trmal.net/resources/83/midlands_english_male.zip
wget https://openslr.trmal.net/resources/83/northern_english_female.zip
wget https://openslr.trmal.net/resources/83/northern_english_male.zip
wget https://openslr.trmal.net/resources/83/scottish_english_female.zip
wget https://openslr.trmal.net/resources/83/scottish_english_male.zip
wget https://openslr.trmal.net/resources/83/southern_english_female.zip
wget https://openslr.trmal.net/resources/83/southern_english_male.zip
wget https://openslr.trmal.net/resources/83/welsh_english_female.zip
wget https://openslr.trmal.net/resources/83/welsh_english_male.zip


# unzipt all zip dataset to 02_process
unzip '*.zip' -d 02_process

# step 1, add speaker, dialect types and gender information
python3 01_process_line_index_all.py

# step 2, add files route to csv
python3 02_process_line_index_all_add_route.py

#step3, split into train, dev and test
python3 03_process_line_index_all_split.py

cd ../..
python3 -m nkululeko.resample --config data/en-dialect/en-dialect_exp.ini
python3 -m nkululeko.nkululeko --config data/en-dialect/en-dialect_exp.ini
```

Reference:  
[1] <https://openslr.org/83/>  
[2] https://storage.googleapis.com/gweb-research2023-media/pubtools/5579.pdf 
