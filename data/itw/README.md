# Nkululeko pre-processing for ITW dataset

## Description
This directory contains the pre-processing scripts and instructions for preparing the ITW (In The Wild) [1,2] dataset using the Nkululeko framework. 

# Commands 

```bash
wget https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip
unzip release_in_the_wild.zip
python process_database.py
cd ../..
python -m nkululeko.nkululeko --config data/itw/exp_focal.ini
```

Reference:  
[1] N. M. Müller, P. Czempin, F. Dieckmann, A. Froghyar, and K. Böttinger, “Does Audio Deepfake Detection Generalize?,” Proc. Annu. Conf. Int. Speech Commun. Assoc. INTERSPEECH, vol. 2022-Septe, no. September, pp. 2783–2787, 2022, doi: 10.21437/Interspeech.2022-108.  
[2]  https://deepfake-total.com/in_the_wild  