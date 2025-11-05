# Nkululeko Copilot Instructions

## Repository Overview

**Nkululeko** is a Python framework for detecting speaker characteristics through machine learning experiments. It orchestrates data loading, feature extraction, and model training for speech processing tasks like emotion, age, gender, and disorder detection. The framework allows users to configure experiments via INI files without directly coding in Python.

**Repository Size**: ~500+ files across data, examples, and source code  
**Languages**: Python (primary), Shell scripts  
**Python Version**: 3.9+ (tested on 3.10, 3.11, 3.12, 3.13)  
**Key Dependencies**: scikit-learn, xgboost, pandas, numpy, audiofile, opensmile, transformers  
**Optional Dependencies**: PyTorch, TensorFlow, Spotlight (Python 3.13 not supported for Spotlight)  

## Installation & Setup

### Standard Installation

**ALWAYS** use virtual environments to avoid dependency conflicts:

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate.bat

# Install from source (for development)
pip install -e .

# OR install test dependencies (lighter, no heavy ML frameworks)
pip install -r requirements-test.txt
pip install -e .
```

### Important Installation Notes

1. **PyPI timeout issues**: If pip install times out, retry or increase timeout: `pip install --timeout=300 -e .`
2. **PyTorch**: For CPU-only (recommended for CI): `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
3. **System dependencies**: Install `sox` and `libportaudio2` on Ubuntu: `sudo apt-get install -y sox libportaudio2`
4. **Python 3.13+**: Spotlight dependencies are not supported. Skip with conditional checks.
5. **Dependencies order**: Install PyYAML first when using Spotlight to avoid dependency issues

## Build, Test, and Validation

### Running Tests

**Primary test command** (from repository root):

```bash
python tests/test_install.py
```

This creates a virtual environment in `build/` directory, installs nkululeko, and runs module import tests.

**For specific Python versions**:

```bash
python tests/test_install.py --python 3.11
```

**Direct module tests** (requires installed package):

```bash
python -m unittest tests/test_modules.py
```

**Note**: Tests create a `build/` directory that persists. Delete it manually to start fresh.

### Linting and Code Formatting

**ALWAYS run these before committing** to pass CI checks:

```bash
# Format code with black
black nkululeko/ --exclude nkululeko/constants.py

# Sort imports
isort --profile black nkululeko/

# Alternatively, use ruff (comprehensive linter)
ruff check --fix --output-format=full nkululeko
```

**Important**: `nkululeko/constants.py` is excluded from black formatting (contains VERSION string).

### Running Experiments

Basic experiment execution:

```bash
python -m nkululeko.nkululeko --config <config_file.ini>
```

Example configurations are in the `examples/` directory. The main workflow downloads test datasets (e.g., emodb from Zenodo) before running experiments.

### CI Workflows

**All CI workflows are in `.github/workflows/`**:

1. **format_code.yml**: Checks black code formatting (excludes constants.py)
2. **isort.yaml**: Validates import sorting with `--profile black`
3. **py310-aud-csv.yml**: Full integration test with Python 3.10
   - Downloads emodb dataset from Zenodo
   - Runs experiments with opensmile features + SVM/XGB
   - Checks for "DONE" in output (approx. 5-10 minutes)
4. **py311.yml, py312.yml, py313.yml**: Basic installation tests

**Note**: The py310 workflow includes disk space cleanup to free ~14GB:

```bash
sudo rm -rf /usr/share/dotnet /opt/ghc /usr/local/share/boost "$AGENT_TOOLSDIRECTORY"
```

## Project Architecture

### Directory Structure

```
nkululeko/
├── nkululeko/              # Main package
│   ├── nkululeko.py        # Main experiment entry point
│   ├── constants.py        # VERSION and global constants
│   ├── experiment.py       # Core Experiment class
│   ├── models/             # ML model implementations (SVM, XGB, MLP, CNN, etc.)
│   ├── feat_extract/       # Feature extractors (opensmile, praat, wav2vec2, etc.)
│   ├── data/               # Data loading and preprocessing
│   ├── augmenting/         # Audio augmentation
│   ├── reporting/          # Result reporting and visualization
│   ├── utils/              # Utility functions
│   ├── demo.py             # Demo/inference module
│   ├── test.py            # Test set evaluation
│   ├── explore.py          # Data exploration
│   ├── augment.py          # Augmentation module
│   ├── ensemble.py         # Ensemble learning
│   ├── multidb.py          # Multi-database comparison
│   └── optim.py            # Hyperparameter optimization
├── data/                   # Dataset definitions (60+ datasets)
│   ├── emodb/              # Berlin emotional database configs
│   ├── polish/             # Polish speech emotion dataset
│   └── ...
├── examples/               # Example INI configuration files
├── tests/                  # Test suite
│   ├── test_install.py     # Installation verification
│   └── test_modules.py     # Module import tests
├── scripts/                # Helper scripts for running experiments
├── docs/                   # Sphinx documentation
├── pyproject.toml          # Modern Python packaging config
├── setup.py                # Legacy setup script (reads from constants.py)
├── setup.cfg               # Additional setup configuration
└── requirements.txt        # Core dependencies
```

### Key Files

- **nkululeko/constants.py**: Contains `VERSION = "1.0.1"` - update for releases
- **pyproject.toml**: Modern build system configuration, dependencies, scripts
- **setup.py**: Reads VERSION from constants.py, defines entry points
- **ini_file.md**: Comprehensive documentation of INI configuration options

### Configuration System

Experiments are defined in **INI files** with sections:
- `[EXP]`: Experiment settings (name, root, runs, epochs)
- `[DATA]`: Datasets, paths, split strategies, label mappings
- `[FEATS]`: Feature extractors (os=opensmile, praat, wav2vec2, etc.)
- `[MODEL]`: ML model type (svm, xgb, mlp, cnn, knn, tree)
- `[PLOT]`: Visualization options
- `[EXPL]`: Explainability (SHAP, feature importance)

See `ini_file.md` for complete documentation.

### Entry Points (Modules)

All modules accept `--config <file.ini>`:

- **nkululeko.nkululeko**: Main experiment runner
- **nkululeko.demo**: Demo trained models
- **nkululeko.test**: Evaluate on test sets
- **nkululeko.explore**: Data exploration/visualization
- **nkululeko.augment**: Audio augmentation
- **nkululeko.ensemble**: Ensemble/late fusion
- **nkululeko.multidb**: Multi-database experiments
- **nkululeko.predict**: Predict with pre-trained models
- **nkululeko.optim**: Hyperparameter optimization
- **nkululeko.resample**: Check/fix sampling rates
- **nkululeko.segment**: Voice activity detection

## Common Issues & Workarounds

### Installation Issues

1. **Timeout during pip install**: Network issues with PyPI are common. Retry or use `--timeout=300`.
2. **Conflicting dependencies**: Always use a fresh virtual environment.
3. **PyTorch CUDA vs CPU**: Default installs GPU version. For CI/testing, use CPU-only build.
4. **Spotlight on Python 3.13+**: Will fail. Check Python version and skip if >= 3.13.

### Running Experiments

1. **Missing datasets**: Most example configs expect datasets in `data/<name>/` directory. The CI workflow downloads them; local runs may need manual download.
2. **Disk space**: ML experiments generate large artifacts (models, features, plots) in `exp_*/` and `results/` folders. CI workflow cleans disk space first.
3. **Long runtimes**: Full experiments take 5-10 minutes. Use timeouts of at least 300 seconds for test commands.
4. **Output validation**: Check for "DONE" string in stdout to confirm successful completion.

### Code Changes

1. **Black formatting**: Exclude `nkululeko/constants.py` from formatting to preserve VERSION string format.
2. **Import order**: Use `isort --profile black` to match black's style.
3. **Version updates**: Update `nkululeko/constants.py` VERSION string for releases.

## Development Workflow

1. **Clone and setup**:
   ```bash
   git clone https://github.com/felixbur/nkululeko.git
   cd nkululeko
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements-test.txt
   pip install -e .
   ```

2. **Make changes**: Edit code in `nkululeko/` directory.

3. **Format and lint**:
   ```bash
   black nkululeko/ --exclude nkululeko/constants.py
   isort --profile black nkululeko/
   ```

4. **Test**:
   ```bash
   python tests/test_install.py
   # OR for quick check:
   python -m unittest tests/test_modules.py
   ```

5. **Run example experiment** (if datasets available):
   ```bash
   python -m nkululeko.nkululeko --config examples/exp_emodb_os_svm.ini
   ```

## Critical Commands Summary

**Installation**: `pip install -e .` (after activating venv)  
**Test**: `python tests/test_install.py`  
**Format**: `black nkululeko/ --exclude nkululeko/constants.py`  
**Sort imports**: `isort --profile black nkululeko/`  
**Run experiment**: `python -m nkululeko.nkululeko --config <config.ini>`  

## Trust These Instructions

These instructions are comprehensive and validated against the repository structure, CI workflows, and documentation. Only search further if:
- Instructions are incomplete for your specific task
- You encounter errors not documented here
- You need dataset-specific details not covered

Refer to `README.md`, `CONTRIBUTING.md`, and `ini_file.md` for additional details.
