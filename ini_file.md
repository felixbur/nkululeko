# Overview of options for the nkululeko framework

* To be specified in a .ini file, [config parser syntax](https://zetcode.com/python/configparser/)
* Kind of all (well, most) values have defaults

## Contents

- [Overview of options for the nkululeko framework](#overview-of-options-for-the-nkululeko-framework)
  - [Contents](#contents)
  - [Sections](#sections)
    - [EXP](#exp)
    - [DATA](#data)
    - [AUGMENT](#augment)
    - [SEGMENT](#segment)
    - [FEATS](#feats)
    - [MODEL](#model)
    - [EXPL](#expl)
    - [PREDICT](#predict)
    - [EXPORT](#export)
    - [CROSSDB](#crossdb)
    - [PLOT](#plot)
    - [RESAMPLE](#resample)
    - [REPORT](#report)

## Sections

### EXP

* **root**: experiment root folder
  * root = ./results/
* **type**: the kind of experiment
  * type = classification
  * possible values:
    * **classification**: supervised learning experiment with a restricted set of categories (e.g., emotion categories).
    * **regression**: supervised learning experiment with continuous values (e.g., speaker age in years).
* **store**: (relative to *root*) folder for caches
  * store = ./store/
* **name**: a name for debugging output
  * name = emodb_exp
* **fig_dir**: (relative to *root*) folder for plots
  * fig_dir = ./images/
* **res_dir**: (relative to *root*) folder for result output
  * res_dir = ./results/
* **models_dir**: (relative to *root*) folder to save models
  * models_dir = ./models/
* **runs**: number of runs (e.g., to average over random initializations)
  * runs = 1
* **epochs**: number of epochs for ANN training
  * epochs = 1
* **save**: save the experiment as a pickle file to be restored again later (True or False)
  * save = False
* **save_test**: save the test predictions as a new database in CSV format (default is False)
  * save_test = ./my_saved_test_predictions.csv
* **databases**: name of databases to compare for the *multidb* module
  * databases = ['emodb', 'timit']
* **use_splits**: can be used for multidb module to use the orginal split sets when train or test database. Else the whole database is used.
  * use_splits = True
### DATA

* **type**: just a flag now to mark continuous data, so it can be binned to categorical data (using *bins* and *labels*)
  * type = continuous
* **databases**: list of databases to be used in the experiment
  * databases = ['emodb', 'timit']
* **tests**: Datasets to be used as test data for the given best model. The databases do NOT have to appear in the **databases** field!
  * tests = ['emovo']
* **root_folders**: specify an additional configuration specifically for all entries starting with a dataset name, acting as global defaults.
  * root_folders = data_roots.ini
* **db_name**: path with audformatted repository for each database listed in 'databases*. If this path is not absolute, it will be treated relative to the experiment folder.
  * emodb = /home/data/audformat/emodb/
* **db_name.type**: type of storage, e.g., audformat database or 'csv' (needs header: file, speaker, task)
  * emodb.type = audformat
* **db_name.absolute_path**: only for 'csv' databases: are the audio file paths relative or absolute? If not absolute, they will be treated relative to the database parent folder. NOT the experiment root folder.
  * my_data.absolute_path = True
* **db_name.audio_path**: only for 'csv' databases: are the audio files in a special common folder?
  * my_data.audio_path = wav_files/
* **db_name.mapping**: mapping python dictionary to map between categories for cross-database experiments (format: {'target_emo':'source_emo'})
  * emodb.mapping = {'anger':'angry', 'happiness':'happy', 'sadness':'sad', 'neutral':'neutral'}
* **db_name.colnames**: mapping to rename columns to standard names
  * my_data.colnames = {'speaker':'Participant ID', 'sex':'gender', 'Age': 'age'}
* **db_name.split_strategy**: How to identify sets for train/development data splits within one database
  * emodb.split_strategy = reuse
  * Possible values:
    * **database**: default (*task*.train and *task*.test)
    * **specified**: specify the tables (an opportunity to assign multiple or no tables to train or dev set)
      * emodb.train_tables = ['emotion.categories.train.gold_standard']
      * emodb.test_tables = ['emotion.categories.test.gold_standard']
    * **speaker_split**: split samples randomly but speaker disjunct, given a percentage of speakers for the test set.
      * emodb.test_size = 50 (default:20)
    * **list of test speakers**: you can simply provide a list of test ids
      * emodb.split_strategy = [12, 14, 15, 16]
    * **random**: split samples randomly (but NOT speaker disjunct, e.g., no speaker info given or each sample a speaker), given a percentage of samples for the test set.
      * emodb.tests_size = 50 (default:20)
    * **reuse**: reuse the splits after a *speaker_split* run to save time with feature extraction.
    * **train**: use the entire database for training
    * **test**: use the entire database for evaluation
* **db_name.target_tables**: tables that contain the target / speaker / sex labels
  * emodb.target_tables = ['emotion']
* **db_name.files_tables**: tables that contain the audio file names
  * emodb.files_tables = ['files']
* **db_name.test_tables**: tables that should be used for testing
  * emodb.test_tables = ['emotion.categories.test.gold_standard']
* **db_name.train_tables**: tables that should be used for training
  * emodb.train_tables = ['emotion.categories.train.gold_standard']
* **db_name.limit_samples**: maximum number of random N samples per table (for testing with very large data mainly)
  * emodb.limit_samples = 20
* **db_name.required**: force a data set to have a specific feature (for example, filter all sets that have gender labeled in a database where this is not the case for all samples, e.g. MozillaCommonVoice)
  * emodb.required = gender
* **db_name.limit_samples_per_speaker**: maximum number of samples per speaker (for leveling data where the same speakers have a large number of samples)
  * emodb.limit_samples_per_speaker = 20
* **db_name.min_duration_of_sample**: limit the samples to a minimum length (in seconds)
  * emodb.min_duration_of_sample = 0.0
* **db_name.max_duration_of_sample**: limit the samples to a maximum length (in seconds)
  * emodb.max_duration_of_sample = 0.0
* **db_name.rename_speakers**: add the database name to the speaker names, e.g., because several databases use the same names
  * emodb.rename_speakers = False
* **db_name.filter**: don't use all the data but only selected values from columns: [col, val]*
  * emodb.filter = [['gender', 'female']]
* **db_name.scale**: [scale (standard normalize) the target variable](http://blog.syntheticspeech.de/2024/03/13/nkululeko-how-to-tweak-the-target-variable-for-database-comparison/) (if numeric)
  * my_data.scale = True
* **db_name.reverse**: reverse the target variable (if numeric). I.e. f(x) = abs(x-max)
* **db_name.reverse.max**: max value to be used in the formula above. If omitted, the distribution will start with 0.
* **target**: the task name, e.g. *age* or *emotion*
  * target = emotion
* **labels**: for classification experiments: the names of the categories (is also used for regression when binning the values)
  * labels = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness']
* **bins**: array of integers to be used for binning continuous data
  * bins  = [-100, 40, 50, 60, 70, 100]
* **no_reuse**: don't re-use any tables, but start fresh
  * no_reuse = False
* **min_dur_test**: specify a minimum duration for test samples (in seconds)
  * min_dur_test = 3.5
* **target_divide_by**: divide the target values by some factor, e.g., to make age smaller and encode years from .0 to 1
  * target_divide_by = 100
* **limit_samples**: maximum number of random N samples per sample selection
  * limit_samples = 20
* **limit_samples_per_speaker**: maximum number of samples per speaker per sample selection
  * limit_samples_per_speaker = 20
* **min_duration_of_sample**: limit the samples to a minimum length (in seconds) per sample selection
  * min_duration_of_sample = 0.0
* **max_duration_of_sample**: limit the samples to a maximum length (in seconds) per sample selection
  * max_duration_of_sample = 0.0
* **check_size**: check the filesize of all samples in train and test splits in bytes
  * check_size = 1000
* **check_vad**: check if the files contain speech, using [silero VAD](https://github.com/snakers4/silero-vad)
  * check_vad = True

### AUGMENT

* **augment**: select the methods to augment: either *traditional* or *random_splice*
  * augment = ['traditional', 'random_splice']
  * choices are:
    * *traditional*: uses the [audiomentations package](https://github.com/iver56/audiomentations)
    * *random_splice*: randomly re-orders short splices (obfuscates the words)
* **sample_selection**: select the samples to augment: either *train*, *test*, or *all*
  * sample_selection = all
* **result**: file name to store the augmented data (can then be added to training)
  * result = augmented.csv
* **augmentations**: select the augmentation methods for the audiomentation module. Default provided.
  * augmentations = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.05),Shift(p=0.5),BandPassFilter(min_center_freq=100.0, max_center_freq=6000),])

### SEGMENT

* **sample_selection**: select the samples to segment: either *train*, *test*, or *all*
  * sample_selection = all
* **segment_result**: name of the segmented data table as a result
  * segment_target = segmented.csv
* **method**: select the model
  * method = [silero](https://github.com/snakers4/silero-vad)
* **min_length**: the minimum length of rest samples (in seconds)
  * min_length = 2
* **max_length**: the maximum length of segments; longer ones are cut here.  (in seconds)
  * max_length = 10

### FEATS

* **type**: a comma-separated list of types of features; they will be column-wise concatenated
  * type = ['os']
  * possible values:
    * **import**: [already computed features](http://blog.syntheticspeech.de/2022/10/18/how-to-import-features-from-outside-the-nkululeko-software/)
      * **import_file** = pathes to files with features in CSV format
        * import_file = ['path1/file1.csv', 'path2/file1.csv2']  
    * **mld**: [mid-level-descriptors](http://www.essv.de/paper.php?id=447)
      * **mld.model** = *path to the mld sources folder*
      * **min_syls** = *minimum number of syllables*
    * **os**: [open smile features](https://audeering.github.io/opensmile-python/)
      * **set** = eGeMAPSv02 *(features set)*
      * **level** = functionals *(or lld: feature level)*
      * **os.features**: list of selected features (disregard others)
    * **praat**: Praat selected features thanks to [David R. Feinberg scripts](https://github.com/drfeinberg/PraatScripts)
      * **praat.features**: list of selected features (disregard others)
    * **spectra**: Melspecs for convolutional networks
      * **fft_win_dur** = 25 *(msec analysis frame/window length)*
      * **fft_hop_dur** = 10 *(msec hop duration)*
      * **fft_nbands** = 64 *(number of frequency bands)*
    * **ast**: [audio spectrogram transformer](https://arxiv.org/abs/2104.01778) features from MIT
    * **trill**: [TRILL embeddings](https://ai.googleblog.com/2020/06/improving-speech-representations-and.html) from Google
      * **trill.model** = *path to the TRILL model folder, optional*
    * **wav2vec variants**: [wav2vec2 embeddings](https://huggingface.co/facebook/wav2vec2-large-robust-ft-swbd-300h) from facebook
      * "wav2vec2-large-robust-ft-swbd-300h"
      * **wav2vec.model** = *path to the wav2vec2 model folder*
      * **wav2vec2.layer** = *which last hidden layer to use*
    * **Hubert variants**: [facebook Hubert models](https://ai.meta.com/blog/hubert-self-supervised-representation-learning-for-speech-recognition-generation-and-compression/)
      * "hubert-base-ls960", "hubert-large-ll60k", "hubert-large-ls960-ft", hubert-xlarge-ll60k, "hubert-xlarge-ls960-ft"
    * **WavLM**:
      * "wavlm-base", "wavlm-base-plus", "wavlm-large"
    * **Whisper**: [whisper models](https://huggingface.co/models?other=whisper)
      * "whisper-base", "whisper-large", "whisper-medium", "whisper-tiny"
    * **audmodel**: [audEERING emotion model embeddings](https://arxiv.org/abs/2203.07378), wav2vec2.0 model finetuned on [MSPPodcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) emotions, embeddings
      * **aud.model** = ./audmodel/ (*path to the audEERING model folder*)
    * **auddim**: [audEERING emotion model dimensions](https://arxiv.org/abs/2203.07378), wav2vec2.0 model finetuned on [MSPPodcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html) arousal, dominance, valence
    * **agender**: [audEERING age and gender model embeddings](https://arxiv.org/abs/2306.16962), wav2vec2.0 model finetuned on [several age databases](https://github.com/audeering/w2v2-age-gender-how-to), embeddings
      * **agender.model** = ./agender/ (*path to the audEERING model folder*)
    * **agender_agender**: [audEERING age and gender model age and gender predictions](https://arxiv.org/abs/2306.16962), wav2vec2.0 model finetuned on [several age and gendeer databases](https://github.com/audeering/w2v2-age-gender-how-to): age, female, male, child
    * **clap**: [Laion's Clap embedding](https://github.com/LAION-AI/CLAP)
    * **xbow**: [open crossbow](https://github.com/openXBOW) features codebook computed from open smile features
      * **xbow.model** = *path to xbow root folder (containing xbow.jar)*
      * **size** = 500 *(codebook size, rule of thumb: should grow with datasize)*
      * **assignments** = 10 *(number of words in the bag representation where the counter is increased for each input LLD, rule of thumb: should grow/shrink with codebook size)*
    * **snr**: estimated SNR (signal-to-noise ratio)
    * **mos**: estimated [MOS](https://arxiv.org/pdf/2304.01448.pdf) (mean opinion score)
    * **pesq**: estimated [PESQ](https://arxiv.org/pdf/2304.01448.pdf) (Perceptual Evaluation of Speech Quality)
    * **sdr**: estimated [SDR](https://arxiv.org/pdf/2304.01448.pdf) (Perceptual Evaluation of Speech Quality)
    * **spkrec**: speaker-id: [speechbrain embeddings](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb)
    * **stoi**: estimated [STOI](https://arxiv.org/pdf/2304.01448.pdf) (Perceptual Evaluation of Speech Quality)
* **features** = *python list of selected features to be used (all others ignored)*
  * features = ['JitterPCA', 'meanF0Hz', 'hld_sylRate']
* **no_reuse**: don't re-use already extracted features, but start fresh
  * no_reuse = False
* **store_format**: how to store the features: possible values [pkl | csv]
  * store_format = pkl
* **scale**: scale the features
  * scale=standard
  * possible values:
    * **standard**: z-transformation (mean of 0 and std of 1) based on the training set
    * **robust**: robust scaler
    * **speaker**: like *standard* but based on individual speaker sets (also for the test)
    * **bins**: convert feature values into 0, .5 and 1 (for low, mid and high)
* **set**: name of opensmile feature set, e.g. eGeMAPSv02, ComParE_2016, GeMAPSv01a, eGeMAPSv01a
  * set = eGeMAPSv02
* **level**: level of opensmile features
  * level = functional
  * possible values:
    * **functional**: aggregated over the whole utterance
    * **lld**: low-level descriptor: framewise
* **balancing**: balance the features with respect to [class distribution](https://imbalanced-learn.org/stable/)
  * balancing=smote
  * possible values:
    * **ros**: simply repeat random samples from the minority classes
    * **smote**: *invent* new minority samples by little changes from the existing ones
    * **adasyn**: similar to smote, but resulting in uneven class distributions

### MODEL

* **type**: type of classifier
  * type = svm
  * possible values:
    * **bayes**: Naive Bayes classifier
    * **cnn**: Convolutional neural network (only works with feature type=spectra)
    * **finetune**: Finetune a transformer model with [huggingface](https://huggingface.co/docs/transformers/training). In this case the features are ignored, because audiofiles are used directly.
      * **pretrained_model**: Base model for finetuning/transfer learning. Variants of wav2vec2, Hubert, and WavLM are tested to work. Default is facebook/wav2vec2-large-robust-ft-swbd-300h.
        * pretrained_model = microsoft/wavlm-base
      * **push_to_hub**: For finetuning, whether to push the model to the huggingface model hub. Default is False.
        * push_to_hub = True
      * **max_duration**: Max. duration of samples/segments for the transformer in seconds, frames are pooled.
        * max_duration = 8.0
    * **gmm**: Gaussian mixture classifier
      * GMM_components = 4
      * GMM_covariance_type = [full | tied | diag | spherical](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html)
    * **knn**: k nearest neighbor classifier
      * K_val = 5
      * KNN_weights = uniform | distance
    * **knn_reg**: K nearest neighbor regressor
    * **mlp**: Multi-Layer-Perceptron for classification
    * **mlp_reg**: Multi-Layer-Perceptron for regression
    * **svm**: Support Vector Machine
      * C_val = 0.001
      * kernel = rbf # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    * **svr**: Support Vector Regression
      * C_val = 0.001
      * kernel = rbf # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    * **tree**: Classification tree classifier
    * **tree_reg**: Classification tree regressor
    * **xgb**: XG-Boost
    * **xgr**: XG-Boost Regression
* **tuning_params**: possible tuning parameters for x-fold optimization (for Bayes, KNN, KNN_reg, Tree, Tree_reg, SVM, SVR, XGB and XGR)
  * tuning_params = ['subsample', 'n_estimators', 'max_depth']
    * subsample = [.5, .7]
    * n_estimators = [50, 80, 200]
    * max_depth = [1, 6]
* **scoring**: scoring measure for the optimization
  * scoring = recall_macro
* **layers**: layer outline (number of hidden layers and number of neurons per layer) for the MLP as a python dictionary
  * layers = {'l1':8, 'l2':4}
* **class_weight**: add class_weight to the linear classifier (XGB, SVM) fit methods for imbalanced data (True or False)
  * class_weight = False
* **logo**: leave-one-speaker group out. Will disregard train/dev splits and split the speakers in *logo* groups and then do a LOGO evaluation. If you want LOSO (leave one speaker out), simply set the number to the number of speakers.
  * logo = 10
* **k_fold_cross**: k-fold-cross validation. Will disregard train/dev splits and do a stratified cross validation (meaning that classes are balanced across folds). speaker id is ignored.
  * k_fold_cross = 10
* **loss**: A loss function for regression ANN models (classification models use Cross Entropy Loss with or without class weights)
  * loss = mse/cross
  * possible values (SHOULD correspond with *measure*):
    * **mse**: mean squared error
    * **mae**: mean average error
    * **1-ccc**: concordance correlation coefficient
    * **cross**: cross entropy correlation
    * **f1**: Soft (differentiable) F1 Loss
* **measure**: A measure/metric to report progress with regression experiments (classification is UAR)
  * measure = mse
  * possible values:
    * **mse**: mean squared error
    * **mae**: mean absolute error
    * **ccc**: concordance correlation coefficient
* **learning_rate**: The learning rate for ANN models
  * learning_rate = 0.0001
* **drop**: Adding dropout (after each hidden layer). Value states dropout probability
  * drop = .5
* **batch_size**: Size of the batch before backpropagation for neural nets
  * batch_size = 8
* **num_workers**: Number of parallel processes for neural nets
  * num_workers = 5
* **device**: For torch/huggingface models: select your GPU number if you have one. Values are either "cpu" or GPU ids (e.g., 0, 1 or both "0,1"). By default, the GPU/CUDA is used if available, otherwise is CPU.
  * device = 0
* **patience**: Number of epochs to wait if the result gets better (for early stopping)
  * patience = 5

### EXPL

* **model**: Which model to use to estimate feature importance.
  * model = ['log_reg'] # can be all models from the [MODEL](#model) section, If they are combined, the mean result is used.
* **max_feats**: Maximal number of important features
  * max_feats = 10
* **sample_selection**: Which sample set/split to use for feature importance, sample distribution, spotlight and feature distributions
  * sample_selection = all # either all, train or test
* **feature_distributions**: plot distributions for features and analyze importance
  * feature_distributions = True
* **permutation**: use [feature permutation](https://scikit-learn.org/stable/modules/permutation_importance.html) to determine the best features. Make sure to test the models before.
  * permutation = True
* **scatter**: make a scatter plot of combined train and test data, colored by label.
  * scatter = ['tsne', 'umap', 'pca']
* **scatter.target**: target for the scatter plot (defaults to *target* value).
  * scatter = ['age', 'gender', 'likability]
* **scatter.dim**: dimension of reduction, can be 2 or 3.
  * scatter.dim = 2
* **plot_tree**: Plot a decision tree for classification (Requires model = tree)
  * plot_tree = False
* **value_counts**: plot distributions of target for the samples and speakers (in the *image_dir*)
  * value_counts = [['gender'], ['age'], ['age', 'duration']]
* **column.bin_reals**: If the column variable is real numbers (instead of categories), should it be binned? for any value in *value_counts* as well as the target variable
  * age.bin_reals = True
* **dist_type**: type of plot for value counts, either histogram or density estimation (kde)
  * dist_type = hist
* **spotlight**: open a web-browser window to inspect the data with the [spotlight software](https://github.com/Renumics/spotlight). Needs package *renumics-spotlight* to be installed!
  * spotlight = False
* **shap**: comopute [SHAP](https://shap.readthedocs.io/en/latest/) values
  * shap = False

### [PREDICT](#predict)

* **targets**: Speaker/speech characteristics to be predicted by some models
  * targets = ['gender', 'age', 'snr', 'arousal', 'valence', 'dominance', 'pesq', 'mos']
* **sample_selection**: which split: [train, test, all]
  * sample_selection = all

### EXPORT

* **target_root**: New root directory for the database, will be created
  * target_root = ./exported_data/
* **orig_root**: Path to folder that is parent to the original audio files
  * orig_root = ../data/emodb/wav
* **data_name**: Name for the CSV file
  * data_name = exported_database
* **segments_as_files**: Whether original files should be used, or segments split (resulting potentially in many new files).
  * segments_as_files = False

### CROSSDB

* **train_extra**: add a additional training partition to all experiments in [the cross database series](http://blog.syntheticspeech.de/2024/01/02/nkululeko-compare-several-databases/). This extra data should be described [in a root_folders file](http://blog.syntheticspeech.de/2022/02/21/specifying-database-disk-location-with-nkululeko/)
  * train_extra = ['addtrain_db_1', 'addtrain_db_2']

### PLOT

* **name**: special name as a prefix for all plots (stored in *img_dir*).
  * name = my_special_config_within_the_experiment
* **epochs**: whether to make a plot each for every epoch result.
  * epochs = False
* **anim_progression**: generate an **animated** GIF from the epoch plots
  * anim_progression = False
* **fps**: frames per second for the animated GIF
  * fps = **1**
* **epoch_progression**: plot the progression of test, train and loss results over epochs
  * epoch_progression = False
* **best_model**: search for the best performing model and plot conf matrix (needs *MODEL.store* to be turned on)
  * best_model = False
* **combine_per_speaker**: print an extra confusion plot where the predictions per speaker are combined, with either the `mode` or the `mean` function
  * combine_per_speaker = mode
* **format**: format for plots, either *png* or *eps* (for scalable graphics)
  * format = png

### RESAMPLE

* **sample_selection**: which split: [train, test, all]
  * sample_selection = all
* **replace**: whether samples should be replaced right where they are, or copies done and a new dataframe given
  * replace = False
* **target**: the name of the new dataframe, if replace==false
  * target = data_resampled.csv

### REPORT

* **show**: print the report at the end
  * show = False
* **fresh**: start a new report
* **latex**: generate a latex and PDF document: name of document
  * latex = my_latex_document
* **title**: title for document
* **author**: author for document
