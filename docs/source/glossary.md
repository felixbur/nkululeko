# Glossary of Terms

This glossary provides definitions for key terms and components used in Nkululeko.

## Core Components

1. **Experiment**: Central class orchestrating data loading, feature extraction, and model training (nkululeko/experiment.py)

2. **Dataset**: Base class for loading and managing audio datasets from different sources (nkululeko/data/dataset.py)

3. **Dataset_CSV**: Class handling CSV-based datasets with audio paths and labels (nkululeko/data/dataset_csv.py)

4. **FeatureExtractor**: Orchestrates feature extraction using different extraction methods (nkululeko/feature_extractor.py)

5. **Featureset**: Abstract base class for all feature extraction implementations (nkululeko/feat_extract/featureset.py)

## Feature Extraction Methods

6. **Opensmileset**: Extracts features using OpenSMILE toolkit (nkululeko/feat_extract/feats_opensmile.py)

7. **Wav2vec2**: Extracts embeddings using Facebook's wav2vec2 model (nkululeko/feat_extract/feats_wav2vec2.py)

8. **PraatSet**: Extracts acoustic features using Praat speech analysis software (nkululeko/feat_extract/feats_praat.py)

9. **TRILLset**: Extracts TRILL (Google's Transfer Representation for IL) embeddings (nkululeko/feat_extract/feats_trill.py)

10. **AudmodelSet**: Extracts embeddings from a wav2vec2-based emotion model (nkululeko/feat_extract/feats_audmodel.py)

## Experiment Management

11. **Runmanager**: Manages multiple runs of an experiment, collecting and comparing results (nkululeko/runmanager.py)

12. **Modelrunner**: Manages training and evaluation for a single run with multiple epochs (nkululeko/modelrunner.py)

## Models

13. **Model**: Base class for all machine learning models (nkululeko/models/model.py)

14. **TunedModel**: Fine-tunes pre-trained transformer models like Wav2Vec2 (nkululeko/models/model_tuned.py)

15. **MLPModel**: Multilayer perceptron model for classification tasks (nkululeko/models/model_mlp.py)

16. **SVM_model**: Support Vector Machine model implementation (nkululeko/models/model_svm.py)

17. **XGB_model**: XGBoost model implementation (nkululeko/models/model_xgb.py)

18. **GMM_model**: Gaussian Mixture Model implementation (nkululeko/models/model_gmm.py)

## Reporting and Analysis

19. **Reporter**: Generates evaluation metrics, plots, and reports for experiments (nkululeko/reporting/reporter.py)

20. **ensemble_predictions**: Function combining predictions from multiple models (nkululeko/ensemble.py)

## Audio Processing

21. **Silero_segmenter**: Segments audio using Silero Voice Activity Detection (nkululeko/segmenting/seg_silero.py)

22. **Resampler**: Handles audio resampling to a specified sample rate (nkululeko/augmenting/resampler.py)

## Prediction and Demonstration

23. **TestPredictor**: Handles prediction and result storage for test datasets (nkululeko/test_predictor.py)

24. **Demo_predictor**: Handles real-time or file-based demonstration of trained models (nkululeko/demo_predictor.py)

25. **FeatureAnalyser**: Analyzes feature importance using SHAP and other methods (nkululeko/feat_extract/feats_analyser.py)
