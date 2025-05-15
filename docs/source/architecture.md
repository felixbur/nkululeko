# Architecture and Components

This page describes the architecture and main components of Nkululeko, providing an overview of how the different parts of the system work together.

## Main Files and Directories

- **`nkululeko/`**: Root directory containing the main package code
  - **`nkululeko/nkululeko.py`**: Main entry point for running experiments
  - **`nkululeko/experiment.py`**: Core Experiment class implementation
  - **`nkululeko/feature_extractor.py`**: Handles feature extraction orchestration
  - **`nkululeko/runmanager.py`**: Manages experiment runs and epochs
  - **`nkululeko/data/`**: Dataset handling implementations
  - **`nkululeko/feat_extract/`**: Feature extraction implementations
  - **`nkululeko/models/`**: Machine learning model implementations
  - **`nkululeko/reporting/`**: Reporting and visualization tools
  - **`nkululeko/utils/`**: Utility functions and helpers
  - **`nkululeko/explore.py`**, **`nkululeko/demo.py`**, etc.: Command-line module entry points
- **`tests/`**: Contains configuration files for testing
- **`data/`**: Storage for datasets or symbolic links to datasets
- **`.github/workflows/`**: CI/CD configuration files

## Main Classes and Components

### Experiment

The central class that manages the entire experiment lifecycle:

```python
# nkululeko/experiment.py
class Experiment:
    def __init__(self, config_obj):
        # Initialize experiment from configuration
    def load_datasets(self):
        # Load datasets specified in configuration
    def fill_train_and_tests(self):
        # Split data into training and testing sets
    def extract_feats(self):
        # Extract features from audio files
    def run(self):
        # Execute the experiment runs
```

### FeatureExtractor

Orchestrates feature extraction from audio:

```python
# nkululeko/feature_extractor.py
class FeatureExtractor:
    def __init__(self, data_df, feats_types, data_name, feats_designation):
        # Initialize feature extractor
    def extract(self):
        # Extract features from audio files
    def extract_sample(self, signal, sr):
        # Extract features from a single audio sample
```

### Runmanager

Manages multiple runs of an experiment:

```python
# nkululeko/runmanager.py
class Runmanager:
    def __init__(self, df_train, df_test, feats_train, feats_test, dev_x=None, dev_y=None):
        # Initialize run manager
    def do_runs(self):
        # Execute multiple experiment runs
```

### Model

Base class for machine learning models:

```python
# nkululeko/models/model.py
class Model:
    def __init__(self, df_train, df_test, feats_train, feats_test):
        # Initialize model
    def train(self):
        # Train the model
    def predict(self):
        # Generate predictions
```

### Reporter

Generates reports and visualizations:

```python
# nkululeko/reporting/reporter.py
class Reporter:
    def __init__(self, truths, preds, run, epoch, probas=None):
        # Initialize reporter
    def plot_confmatrix(self, plot_name, epoch=None):
        # Plot confusion matrix
    def print_results(self, epoch=None, file_name=None):
        # Print evaluation results
```

## Command-Line Modules

Each module provides specific functionality:

- `nkululeko.nkululeko`: Main experiment runner
- `nkululeko.explore`: Data and feature exploration
- `nkululeko.demo`: Model demonstration
- `nkululeko.test`: Model testing
- `nkululeko.augment`: Data augmentation
- `nkululeko.ensemble`: Model ensemble creation
- `nkululeko.multidb`: Cross-database experimentation
- `nkululeko.segment`: Audio segmentation

## Data Flow

1. The user creates an INI configuration file specifying the experiment parameters
2. The `Experiment` class loads the configuration and initializes the experiment
3. Datasets are loaded and split into training and testing sets
4. Features are extracted from the audio files using the specified feature extractors
5. The `Runmanager` executes multiple runs of the experiment
6. Models are trained and evaluated
7. Results are reported and visualized

This architecture allows for a high degree of flexibility and extensibility, enabling users to experiment with different combinations of datasets, features, and models without having to write extensive code.
