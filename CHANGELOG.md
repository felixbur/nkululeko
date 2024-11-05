Changelog
=========

Version 0.91.3
--------------
* some additions for robustness

Version 0.91.2
--------------
* making lint work by excluding constants from check

Version 0.91.1
--------------
* minor refactoring in ensemble module

Version 0.91.0
--------------
* fixed duration display in segmentation
* added possibility to use original segmentations (without max. duration)

Version 0.90.4
--------------
* added plot format for multidb

Version 0.90.3
--------------
* refactorings and documentations

Version 0.90.2
--------------
* added probability output to finetuning classification models
* switched path to prob. output from "store" to "results"

Version 0.90.1
--------------
* Add balancing for finetune and update data README

Version 0.90.0
--------------
* augmentation can now be done without target
* random splicing params configurable
* made kde default for plot continous/categorical plots

Version 0.89.2
--------------
* fix shap value calculation
  
Version 0.89.1
--------------
* print and save result of feature importance

Version 0.89.0
--------------
* added Roc plots and classification report on Debug


Version 0.88.12
---------------
* added n_jobs for sklearn processing
* re_named num_workers n_jobs

Version 0.88.11
--------------
* removed hack in Praat script 

Version 0.88.10
--------------
* SVM C val defaults to 1
* fixed agender_agender naming bug

Version 0.88.9
--------------
* added performance_weighted ensemble

Version 0.88.8
--------------
* some cosmetics

Version 0.88.7
--------------
* added use_splits for multidb

Version 0.88.6
--------------
* added test speaker assign

Version 0.88.5
--------------
* add a unique name to the uncertainty plot
* fix error in speaker embedding (still need speechbrain < 1.0)
* add get_target_name function in util

Version 0.88.4
--------------
* added more ensemble methods, e.g. based on uncertainty

Version 0.88.3
--------------
* fixed bug in false uncertainty estimation
* changed demo live recording

Version 0.88.2
--------------
* changed combine speaker results to show speakers not samples

Version 0.88.1
--------------
* added obligatory scatter plot for regression

Version 0.88.0
--------------
* added ensemble late fusion and AST features

Version 0.87.0
--------------
* added class probability output and uncertainty analysis

Version 0.86.8
--------------
* handle single feature sets as strings in the config

Version 0.86.7
--------------
* handles now audformat tables where the target is in a file index
  
Version 0.86.6
--------------
* now best (not last) result is shown at end

Version 0.86.5
--------------
* fix audio path detection in data csv import

Version 0.86.4
--------------
* add finetuning to the demo module

Version 0.86.3
--------------
* bugfixed: nan in finetuned model and double saving
* import features now get multiindex automatically

Version 0.86.2
--------------
* plots epoch progression for finetuned models now

Version 0.86.1
--------------
* functionality to push to hub
* fixed bug that prevented wavlm finetuning

Version 0.86.0
--------------
* added regression to finetuning
* added other transformer models to finetuning
* added output the train/dev features sets actually used by the model

Version 0.85.2
--------------
* added data, and automatic task label detection

Version 0.85.1
--------------
* fixed bug in model_finetuned that label_num was constant 2

Version 0.85.0
--------------
* first version with finetuning wav2vec2 layers

Version 0.84.1
--------------
* made resample independent of config file

Version 0.84.0
--------------
* added SHAP analysis
* started with finetuning

Version 0.83.3
--------------
* fixed a naming error in trill features that prevented storage of experiment

Version 0.83.2
--------------
* added default cuda if present and not stated

Version 0.83.1
--------------
* add test module to nkuluflag

Version 0.83.0
--------------
* test module now prints out reports

Version 0.82.4
--------------
* fixed bug in wavlm 

Version 0.82.3
--------------
* fixed another audformat peculiarity to interprete time values as nanoseconds

Version 0.82.2
--------------
* fixed audformat peculiarity that dataframes can have only one column

Version 0.82.1
--------------
* Add more test for GC action

Version 0.82.0
--------------
* added nkuluflag module

Version 0.81.7
--------------
* bugfixes
* added whisper feature extractor

Version 0.81.6
--------------
* updated documentation
* updated crema-d
* updated tests

Version 0.81.5
--------------
* added sex=gender for speaker mappings

Version 0.81.4
--------------
* fixed bug in demo module
* removed [MODEL] save

Version 0.81.3
--------------
* added confidence intervals to result reporting

Version 0.81.2
--------------
* added a parselmouth.Praat error if pitch out of range
* changed file path for demo_predictor

Version 0.81.1
--------------
* fixed bugs in demo module 
* made kernel for SVM/SVR configurable

Version 0.81.0
--------------
* added test selection to test module

Version 0.80.4
--------------
* added test-file folder to demo file lists

Version 0.80.3
--------------
* made sounddevice use optional as Portaudio library causes difficulties

Version 0.80.2
--------------
* fixed bug that caused clash with GPU/CPU use

Version 0.80.1
--------------
* added support for string value in import_features
+ added support for multiple extra training databases when doing multi-db experiments

Version 0.80.0
--------------
* fixed bug no feature import
* add support for multiple import feature files

Version 0.79.5
--------------
* fixed bug on demo without in- or output
* fixed bug that demo with DL feature extractors did not work

Version 0.79.4
--------------
* added functionality in demo for regression

Version 0.79.3
--------------
* fixed bug that test module did not work
* fixed bug that demo module did not work for ANNs
* added csv output for demo mode and file lists

Version 0.79.2
--------------
* fixed bug and report number of epochs for early stopping 

Version 0.79.1
--------------
* root directory does not have to end with /

Version 0.79.0
--------------
* added extra_train for multidb experiment

Version 0.78.2
--------------
* added transformer layer selection for wav2vec2
* removed best_model and epoch progression for non-DL models

Version 0.78.1
--------------
* added evaluation loss

Version 0.78.0
--------------
* added 3-d scatter plots
* removed epoch-plots if epoch_num=1

Version 0.77.14
--------------
* fixed bug preventing bin scaling to work

Version 0.77.13
--------------
* added bins scaler

Version 0.77.12
--------------
* fixed bug with scatter plots for numeric targets
* made type of numeric target distributions selectable, default "hist"

Version 0.77.11
--------------
* added simple target distribution plots

Version 0.77.10
--------------
* show the best and not the last result for multidb

Version 0.77.9
--------------
* added results text for multidb

Version 0.77.8
--------------
* added caption to multidb heatmap
* renamed datasets to databases in multidb

Version 0.77.7
--------------
* added multidb module

Version 0.77.6
--------------
* added functions to call modules with config file path directly

Version 0.77.5
--------------
* fixed augmentation bug for python version 10

Version 0.77.4
--------------
* made traditional augmentations (audiomentation module) configurable

Version 0.77.3
--------------
* added augment and train interface

Version 0.77.2
--------------
* added models for features importance computation
  
Version 0.77.1
--------------
* added permutation algorithm to compute feature importance
* shifted util.py to utils

Version 0.77.0
--------------
* added more latex report output
* got splitutils from a package

Version 0.76.0
--------------
* added possibility to aggregate feature importance models

Version 0.75.0
--------------
* added max val for reversing
* added xgb for feature importance

Version 0.74.6
--------------
* added standard Wav2vec2 model

Version 0.74.5
--------------
* added praat feature extractor for one sample

Version 0.74.4
--------------
* fixed bug combining augmentations

Version 0.74.3
--------------
* audiomentations interface changed

Version 0.74.2
--------------
* combined augmentation methods
  
Version 0.74.1
--------------
* fixed various bugs with augmentation

Version 0.74.0 
--------------
* added patience (early stopping)
* added MAE loss and measure

Version 0.73.0
--------------
* added reverse and scale arguments to target variable
* also, the data store can now be csv

Version 0.72.0
--------------
* worked over explore value counts section
* added bin_reals for all columns

Version 0.71.4
--------------
* automatic epoch reset if not ANN
* scatter plots now show a regression line

Version 0.71.3
--------------
* enabled scatter plots for all variables

Version 0.71.2
--------------
* enabled scatter plots for continuous labels

Version 0.71.1
--------------
* made a wav2vec default 
* renamed praat features, ommiting spaces
* fixed plot distribution bugs
* added feature plots for continuous targets

Version 0.71.0
--------------
* added explore visuals. 
* all columns from databases should now be usable

Version 0.70.0
--------------
* added imb_learn balancing of training set

Version 0.69.0
--------------
* added CNN model and melspec extractor

Version 0.68.4
--------------
* bugfix: got_gender was uncorrectly set

Version 0.68.3
--------------
* Feinberg Praat scripts ignore error and log filename

Version 0.68.2
--------------
* column names in datasets are now configurable

Version 0.68.1
--------------
* added error message on file to praat extraction

Version 0.68.0
--------------
* added stratification framework for split balancing

Version 0.67.0
--------------
* added first version of spotlight integration

Version 0.66.13
---------------
* small changes related to github worker

Version 0.66.12
---------------
* fixed bug that prevented Praat features to be selected 
  
Version 0.66.11
---------------
* removed torch from automatic install. depends on cpu/gpu machine

Version 0.66.10
---------------
* Removed print statements from feats_wav2vec2

Version 0.66.9
--------------
* Version that should install without requiring opensmile which seems not to be supported by all Apple processors (arm CPU (Apple M1))

Version 0.66.8
--------------
* forgot __init__.py in reporting module

Version 0.66.7
--------------
* minor changes to experiment class

Version 0.66.6
--------------
* minor cosmetics

Version 0.66.5
--------------
* Latex report now with images

Version 0.66.4
--------------
* Pypi version mixup

Version 0.66.3
--------------
* made path to PDF output relative to experiment root

Version 0.66.2
--------------
* enabled data-pacthes with quotes 
* enabled missing category labels
* used tqdm for progress display

Version 0.66.1
--------------
* start on the latex report framework

Version 0.66.0
--------------
* added speechbrain speakerID embeddings 
  
Version 0.65.9
--------------
* added a filter that ensures that the labels have the same size as the features

Version 0.65.8
--------------
* changed default behaviour of resampler to "keep original files"

Version 0.65.7
--------------
* more databases and force wav while resampling

Version 0.65.6
--------------
* minor catch for seaborn in plots

Version 0.65.5
--------------
* added fill_na in plot effect size

Version 0.65.4
--------------
* added datasets to distribution
* changes in wav2vec2

Version 0.65.3
--------------
* various bugfixes

Version 0.65.2
--------------
* fixed bug in dataset.csv that prevented correct paths for relative files
* fixed bug in export module concerning new file directory

Version 0.65.1
--------------
* small enhancements with transformer features

Version 0.65.0
--------------
* introduced export module

Version 0.64.4
--------------
* added num_speakers for reloaded data
* re-formatted all with black

Version 0.64.3
--------------
* added number of speakers shown after data load

Version 0.64.2
--------------
* added __init__.py for submodules

Version 0.64.1
--------------
* fix error on csv

Version 0.64.0
--------------
* added bin_reals
* added statistics for effect size and correlation to plots

Version 0.63.4
--------------
* fixed bug in split selection

Version 0.63.3
--------------
* Introduced data.audio_path


Version 0.63.2
--------------
* re-introduced min and max_length for silero segmenatation

Version 0.63.1
--------------
* fixed bug in resample

Version 0.63.0
--------------
* added wavlm model
* added error on filename for models

Version 0.62.1
--------------
* added min and max_length for silero segmenatation

Version 0.62.0
--------------
* fixed segment silero bug
* added all Wav2vec2 models
* added resampler module
* added error on file for embeddings

Version 0.61.0
--------------
* added HUBERT embeddings
  
Version 0.60.0
--------------
* some bugfixes
* new package structure
* fixed wav2vec2 bugs
* removed "cross_data" strategy 


Version 0.59.1
--------------
* bugfix, after fresh install, it seems some libraries have changed
* added no_warnings
* changed print() to util.debug()
* added progress to opensmile extract
  
Version 0.59.0
--------------
* introduced SQUIM features
* added SDR predict
* added STOI predict

Version 0.58.0
--------------
* added dominance predict
* added MOS predict 
* added PESQ predict 

Version 0.57.0
--------------
* renamed autopredict predict
* added arousal autopredict
* added valence autopredict 


Version 0.56.0
--------------
* added autopredict module
* added snr as feature extractor
* added gender autopredict
* added age autopredict
* added snr autopredict

Version 0.55.1
--------------
* changed error message in plot class

Version 0.55.0
--------------
* added segmentation module

Version 0.54.0
--------------
* added audeering public age and gender model embeddings and age and gender predictions

Version 0.53.0
--------------
* added file checks: size in bytes and voice activity detection with silero

Version 0.52.1
--------------
* bugfix: min/max duration_of_sample was not working

Version 0.52.0
--------------
* added flexible value distribution plots

Version 0.51.0
--------------
* added datafilter

Version 0.50.1
--------------
* added caller information for debug and error messages in Util

Version 0.50.0
--------------
* removed loso and added pre-selected logo (leave-one-group-out), aka folds

Version 0.49.1
--------------
* bugfix: samples selection for augmentation didn't work

Version 0.49.0
--------------
* added random-splicing

Version 0.48.1
--------------
* bugfix: database object was not loaded when dataframe was reused

Version 0.48.0
--------------
* enabled specific feature selection for praat and opensmile features

Version 0.47.1
--------------
* enabled feature storage format csv for opensmile features

Version 0.47.0
--------------
* added praat speech rate features

Version 0.46.0
--------------
* added warnings for non-existent parameters
* added sample selection for scatter plotting

Version 0.45.4
--------------
* added version attribute to setup.cfg

Version 0.45.4
--------------
* added __version__ attribute


Version 0.44.1
--------------
* bugfixing: feature importance: https://github.com/felixbur/nkululeko/issues/23
* bugfixing: loading csv database with filewise index https://github.com/felixbur/nkululeko/issues/24 

Version 0.45.2
--------------
* bugfix: sample_selection in EXPL was required wrongly

Version 0.45.2
--------------
* added sample_selection for sample distribution plots

Version 0.45.1
--------------
* fixed dataframe.append bug

Version 0.45.0
--------------
* added auddim as features
* added FEATS store_format
* added device use to feat_audmodel

Version 0.44.1
--------------
* bugfixes

Version 0.44.0
--------------
* added scatter functions: tsne, pca, umap

Version 0.43.7
--------------
* added clap features

Version 0.43.6
--------------
* small bugs


Version 0.43.5
--------------
* because of difficulties with numba and audiomentations importing audiomentations only when augmenting

Version 0.43.4
--------------
* added error when experiment type and predictor don't match

Version 0.43.3
--------------
* fixed further bugs and added augmentation to the test runs

Version 0.43.2
--------------
* fixed a bug when running continuous variable as classification problem

Version 0.43.1
--------------
* fixed test_runs

Version 0.43.0
--------------
* added augmentation module based on audiomentation

Version 0.42.0
--------------
* age labels should now be detected in databases

Version 0.41.0
--------------
* added feature tree plot

Version 0.40.1
--------------
* fixed a bug: additional test database was not label encoded

Version 0.40.0
--------------
* added EXPL section and first functionality
* added test module (for test databases)

Version 0.39.0
--------------
* added feature distribution plots
* added  plot format

Version 0.38.3
--------------
* added demo mode with list argument

Version 0.38.2
--------------
* fixed a bug concerned with "no_reuse" evaluation

Version 0.38.1
--------------
* demo mode with file argument

Version 0.38.0
--------------
* fixed demo mode

Version 0.37.2
--------------
* mainly replaced pd.append with pd.concat


Version 0.37.1
--------------
* fixed bug preventing praat feature extraction to work

Version 0.37.0
--------------
* fixed bug cvs import not detecting multiindex 

Version 0.36.3
--------------
* published as a pypi module

Version 0.36.0
--------------
* added entry nkululeko.py script


Version 0.35.0
--------------
* fixed bug that prevented scaling (normalization)

Version 0.34.2
--------------
* smaller bug fixed concerning the loss_string

Version 0.34.1
--------------
* smaller bug fixes and tried Soft_f1 loss


Version 0.34.0
--------------
* smaller bug fixes and debug ouputs

Version 0.33.0
--------------
* added GMM as a model type

Version 0.32.0
--------------
* added audmodel embeddings as features

Version 0.31.0
--------------
* added models: tree and tree_reg
  
Version 0.30.0
--------------
* added models: bayes, knn and knn_reg

Version 0.29.2
--------------
* fixed hello world example


Version 0.29.1
--------------
* bug fix for 0.29


Version 0.29.0
--------------
* added a new FeatureExtractor class to import external data

Version 0.28.2
--------------
* removed some Pandas warnings
* added no_reuse function to database.load()

Version 0.28.1
--------------
* with database.value_counts show only the data that is actually used


Version 0.28.0
--------------
* made "label_data" configuration automatic and added "label_result"


Version 0.27.0
--------------
* added "label_data" configuration to label data with trained model (so now there can be train, dev and test set)

Version 0.26.1
--------------
* Fixed some bugs caused by the multitude of feature sets
* Added possibilty to distinguish between absolut or relative pathes in csv datasets

Version 0.26.0
--------------
* added the rename_speakers funcionality to prevent identical speaker names in datasets

Version 0.25.1
--------------
* fixed bug that no features were chosen if not selected

Version 0.25.0
--------------
* made selectable features universal for feature sets

Version 0.24.0
--------------
* added multiple feature sets (will simply be concatenated)

Version 0.23.0
--------------
* added selectable features for Praat interface

Version 0.22.0
--------------
* added David R. Feinberg's Praat features, praise also to parselmouth

Version 0.21.0
--------------

* Revoked 0.20.0
* Added support for only_test = True, to enable later testing of trained models with new test data

Version 0.20.0
--------------

* implemented reuse of trained and saved models

Version 0.19.0
--------------

* added "max_duration_of_sample" for datasets


Version 0.18.6
--------------

* added support for learning and dropout rate as argument


Version 0.18.5
--------------

* added support for epoch number as argument
  
Version 0.18.4
--------------

* added support for ANN layers as arguments

Version 0.18.3
--------------

* added reuse of test and train file sets
* added parameter to scale continous target values: target_divide_by


Version 0.18.2
--------------

* added preference of local dataset specs to global ones
  
Version 0.18.1
--------------

* added regression value display for confusion matrices

Version 0.18.0
--------------

* added leave one speaker group out

Version 0.17.2
--------------

* fixed scaler, added robust



Version 0.17.0
--------------

* Added minimum duration for test samples


Version 0.16.4
--------------

* Added possibility to combine predictions per speaker (with mean or mode function)

Version 0.16.3
--------------

* Added minimal sample length for databases


Version 0.16.2
--------------

* Added k-fold-cross-validation for linear classifiers

Version 0.16.1
--------------

* Added leave-one-speaker-out for linear classifiers


Version 0.16.0
--------------

* Added random sample splits

