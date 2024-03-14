% Nkululeko workshop
% Felix Burkhardt
% April 2023

# Q: What is Nkululeko

A: Framework to perform audio prediction experiments

Combines 

* machine learners (classification and regression) 
* with acoustic features (expert and/or learned)

# Motivation for Nkululeko

* initially own utility collection
* then used as a teaching tool
* can still be used for superficial fast analysis

# Nkululeko setup

steps:

* create virtual python environment
 
    ```
    python -m venv venv
    ```
* activate it
 
    ```
    source venv/bin/activate # linux/mac
    venv\Scripts\Activate.bat # windows
    ```
* install via pip 
  
    ```
    pip install nkululeko
    ```

# Nkululeko docs

online:

* [github](https://github.com/felixbur/nkululeko)
* [blog](http://blog.syntheticspeech.de/2021/08/30/how-to-set-up-your-first-nkululeko-project/)
* [ini file syntax](https://github.com/felixbur/nkululeko/blob/main/ini_file.md)

# Get a database

e.g. [emodb](http://blog.syntheticspeech.de/2021/08/10/get-all-information-from-emodb/)


# Data exploration

Nkululeko functionality is organized in modules

```
python 
    -m nkululeko.explore 
        --config tests/exp_emodb_explore_data_*.ini
```
# Experiment emotion classification

1) Combining praat and opensmile features

    ```
    python 
        -m nkululeko.nkululeko 
            --config tests/exp_emodb_os_praat_xgb.ini
    ```

2) Using learned features (pre-trained and fine-tuned with wav2vec2 transformer architecture)

    ```
    python 
        -m nkululeko.nkululeko 
            --config tests/exp_emodb_audmodel_xgb.ini
    ```

# Experiment age prediction

1) With opensmile features and a [multi-layer perceptron](http://blog.syntheticspeech.de/2022/11/21/different-machine-learners/)

    ```
    python 
        -m nkululeko.nkululeko 
            --config tests/exp_agedb_os_mlp.ini
    ```

2) Using classifyer after binning, with [xgboost](http://blog.syntheticspeech.de/2022/11/21/different-machine-learners/) 

    ```
    python 
        -m nkululeko.nkululeko 
            --config tests/exp_agedb_class_os_xgb.ini
    ```
