# Project Skill: nkululeko

## Overview
Nkululeko is a software to detect speaker characteristics by machine learning experiments with a high-level interface. The idea is to have a framework (based on e.g. sklearn and torch) that can be used to rapidly and automatically analyse audio data and explore machine learning models based on that data.

Some abilities that Nkululeko provides: combines acoustic features and machine learning models (including feature selection and features concatenation); performs data exploration, selection and visualization the results; finetuning; ensemble learning models; soft labeling (predicting labels with pre-trained model); and inference the model on a test set.

Nkululeko orchestrates data loading, feature extraction, and model training, allowing you to specify your experiment in a configuration file. The framework handles the process from raw data to trained model and evaluation, making it easy to run machine learning experiments without directly coding in Python.

## Tech Stack
- Python > 3.10

## Code Conventions
- Add tests for new functions
- Add utility functions to the utils sub-package
- Add docstrings to new functions
- Follow PEP8 style guide

## Project Structure
Nkululeko is organized into several modules:
- nkululeko: main module containing core functionality for data loading, feature extraction, model training, and evaluation.
- explore: module for data exploration and visualization.
- predict: module for making predictions with trained models and existing feature extractors.
- augment: module for data augmentation techniques.
- segment: module for audio segmentation and related utilities.
- flags: module for running experiments with different combinations of parameters specified in configuration files.

All models are in the nkululeko.models subpackage, and all feature extractors are in the nkululeko.features subpackage.

## Result Folder Structure
The results of experiments are stored in a structured way under the root directory specified in the configuration file. The structure is as follows:
- root/
  - experiment_name/
    - images/ # for plots and visualizations
    - models/ # for trained model files
    - logs/ # for training logs and metrics
    - results/ # for evaluation results and metrics
    - store/ # for reusable data like extracted features and predictions
    - cache/ # for temporary files and intermediate results

Usually the result files follow a naming convention that includes the experiment name, dataset names, model type, and relevant parameters for easy identification.

## Documentation
Documentation is under the folder docs/source.

All key-value pairs in the configuration files should be documented in the ini_file.md file, which serves as a reference for users to understand the available options and their effects on the experiments.

## Result Analysis

The results of an experiment are in the results folder. When asked to write an exploration report, compile the statistical results including images in a markdown file, but only if the results are statistically significant. If not, just write a short note about the results and do not include any images.

### Parsing statistical result files

Feature distribution result files are stored as `results/run_0/feat_dist_all_{feature}.txt`. Each file contains lines starting with `overall:` and `pairwise:` that are Python dict literals — parse them with `ast.literal_eval`, do NOT use string grep on the raw text.

Significance levels:
- **highly significant**: p < 0.001
- **significant**: p < 0.05
- **marginally significant**: p < 0.1
- **not significant**: p >= 0.1

A feature is considered significant if:
- the `overall` Kruskal-Wallis result is not "not significant", OR
- at least one pairwise comparison is not "not significant"

The significance field always contains the literal string "not significant" for non-significant results. Checking with `grep -v "not significant"` on a combined string is unreliable because a single string can contain both "not significant" (for some pairs) and "significant" (for others). Always parse with Python and check each entry individually.


## How Claude Should Behave
- Use utility functions from the utils sub-package when possible.
- all pytest calls are allowed
- When the same sequence of statements would appear more than once (e.g. compute stats, then conditionally write two result lines), extract it into a private helper method first and call that helper at each site. Do not inline the same block twice.
