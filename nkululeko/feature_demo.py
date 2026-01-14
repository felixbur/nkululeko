# feature_demo.py
# Demonstration code to extract features from audio files using feature extraction models
"""This script extracts features from audio files using trained feature extraction models.

It can process single files, lists of files, or entire folders and output the features.

Usage:
python -m nkululeko.feature_demo [--config CONFIG] [--file FILE] [--list LIST] [--folder FOLDER] [--outfile OUTFILE] [--model MODEL]

Options:
--config CONFIG     The base configuration file (default: exp.ini)
--file FILE         A file that should be processed (16kHz mono wav)
--list LIST         A file with a list of files, one per line, that should be processed (16kHz mono wav)
--folder FOLDER     A name of a folder where the files within the list are in (default: ./)
--mic               Record audio from microphone for 5 seconds
--outfile OUTFILE   A filename to store the features in CSV format (default: features_output.csv)
--model MODEL       Feature extraction model to use (default: wav2vec2-large-robust-ft-swbd-300h)
                    Options: wav2vec2-*, hubert-*, wavlm-*, whisper-*, ast-*, emotion2vec-*,
                    audmodel (requires audmodel.id in config), opensmile, clap, spkrec, trill, praat, etc.
"""
import argparse
import configparser
import os

import audiofile
import numpy as np
import pandas as pd
import sounddevice as sd
import tempfile

import nkululeko.glob_conf as glob_conf
from nkululeko.constants import VERSION
from nkululeko.utils.util import Util


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from audio files using nkululeko feature extraction models."
    )
    parser.add_argument("--config", default="exp.ini", help="The base configuration")
    parser.add_argument(
        "--file", help="A file that should be processed (16kHz mono wav)"
    )
    parser.add_argument(
        "--list",
        help=(
            "A file with a list of files, one per line, that should be"
            " processed (16kHz mono wav)"
        ),
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--folder",
        help=("A name of a folder where the files within the list are in."),
        nargs="?",
        default="./",
    )
    parser.add_argument(
        "--mic",
        help=("Record audio from microphone for 5 seconds"),
        action="store_true",
    )
    parser.add_argument(
        "--outfile",
        help=("A filename to store the features in CSV format"),
        nargs="?",
        default="features_output.csv",
    )
    parser.add_argument(
        "--model",
        help=(
            "Feature extraction model to use (e.g., wav2vec2-large-robust-ft-swbd-300h,"
            " hubert-large-ll60k, wavlm-base, whisper-base, ast, emotion2vec-base,"
            " audmodel, opensmile, clap, spkrec, trill, praat)"
        ),
        nargs="?",
        default="wav2vec2-large-robust-ft-swbd-300h",
    )
    args = parser.parse_args()

    if args.config is not None:
        config_file = args.config
    else:
        config_file = "exp.ini"

    # test if the configuration file exists
    if not os.path.isfile(config_file):
        print("="*80)
        print(f"ERROR: Configuration file not found: {config_file}")
        print("\nPlease create a config file with the following structure:")
        print("  [EXP]")
        print("  name = feature_demo")
        print("\n  [FEATS]")
        print("  audmodel.id = <model_id>  # Required for audmodel")
        print("  wav2vec2.layer = <layer>  # Optional for wav2vec2")
        print("\n  [MODEL]")
        print("  device = cuda  # or cpu")
        print("="*80)
        exit(1)

    # load configuration
    config = configparser.ConfigParser()
    config.read(config_file)
    print(f"Using configuration from: {config_file}")

    # Initialize global config
    glob_conf.config = config

    module = "feature_demo"
    util = Util(module)
    util.debug(f"running feature extraction demo, nkululeko version {VERSION}")
    util.debug(f"using model: {args.model}")

    # Collect files to process
    files = []
    if args.mic:
        # Record from microphone
        print("Recording from microphone for 5 seconds...")
        print("Start speaking now!")
        duration = 5  # seconds
        sample_rate = 16000  # Hz

        # Record audio
        recording = sd.rec(int(duration * sample_rate),
                          samplerate=sample_rate,
                          channels=1,
                          dtype='float32')
        sd.wait()  # Wait until recording is finished
        print("Recording finished!")

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filename = temp_file.name
        temp_file.close()

        # Write the recording to the temp file
        import soundfile as sf
        sf.write(temp_filename, recording, sample_rate)

        files = [temp_filename]
    elif args.file is not None:
        files = [args.file]
    elif args.list is not None:
        # read audio files from list
        print(f"Reading files from {args.list}")
        list_file = pd.read_csv(args.list, header=None)
        files = list_file.iloc[:, 0].tolist()
        # prepend folder if provided
        if args.folder != "./":
            files = [os.path.join(args.folder, f) for f in files]
    elif args.folder is not None:
        # read audio files from folder
        from nkululeko.utils.files import find_files

        files = find_files(args.folder, relative=False, ext=["wav", "mp3", "flac"])
    else:
        print("ERROR: You must provide --file, --list, --folder, or --mic")
        return

    print(f"Processing {len(files)} file(s)...")

    # Initialize feature extractor based on model type
    feature_extractor = get_feature_extractor(args.model, config)

    # Extract features for all files
    features_list = []

    for i, file in enumerate(files):
        if not os.path.isfile(file):
            print(f"WARNING: File not found: {file}, skipping...")
            continue

        print(f"[{i+1}/{len(files)}] Extracting features from: {file}")

        try:
            # Load audio file using audiofile
            signal, sampling_rate = audiofile.read(file, always_2d=True)

            # Extract features from the loaded audio signal
            features = feature_extractor.extract_sample(signal, sampling_rate)

            # Handle different feature formats
            if isinstance(features, np.ndarray):
                if features.ndim > 1:
                    # Flatten if multi-dimensional
                    features = features.flatten()
            elif isinstance(features, pd.Series):
                features = features.values
            elif isinstance(features, pd.DataFrame):
                features = features.values.flatten()

            features_list.append({"file": file, "features": features})

            # Get feature dimension info on first file
            if i == 0:
                print(f"Feature dimension: {len(features)}")

        except Exception as e:
            print(f"ERROR processing {file}: {str(e)}")
            continue

    # Save features to CSV and print summary
    if len(features_list) > 0:
        # Create DataFrame
        features_array = np.array([f["features"] for f in features_list])
        file_names = [f["file"] for f in features_list]

        # Generate column names
        feature_names = [f"feat_{j}" for j in range(features_array.shape[1])]

        df = pd.DataFrame(features_array, index=file_names, columns=feature_names)

        # Save to CSV
        if args.outfile:
            # Ensure .csv extension
            outfile = args.outfile if args.outfile.endswith('.csv') else args.outfile + '.csv'
            df.to_csv(outfile)
            print(f"\nFeatures saved to {outfile}")

        # Print summary to stdout
        print("\n" + "="*80)
        print("FEATURE EXTRACTION SUMMARY")
        print("="*80)
        print(f"Total files processed: {len(file_names)}")
        print(f"Feature dimension: {features_array.shape[1]}")
        print(f"Output shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
    else:
        print("ERROR: No features were extracted")

    # Clean up temporary file if microphone was used
    if args.mic:
        try:
            os.remove(files[0])
            print(f"Cleaned up temporary recording file")
        except Exception as e:
            print(f"Warning: Could not clean up temporary file: {e}")

    print("\nDONE")


def get_feature_extractor(model_name, config):
    """Get the appropriate feature extractor based on model name.

    Args:
        model_name (str): Name of the feature extraction model
        config (ConfigParser): Configuration object

    Returns:
        Feature extractor instance
    """
    # Import feature extractors
    model_lower = model_name.lower()

    # Determine which extractor to use based on model name
    if "wav2vec2" in model_lower or "wav2vec" in model_lower:
        from nkululeko.feat_extract.feats_wav2vec2 import Wav2vec2

        return Wav2vec2(model_name, config)

    elif "hubert" in model_lower:
        from nkululeko.feat_extract.feats_hubert import Hubert

        return Hubert(model_name, config)

    elif "wavlm" in model_lower:
        from nkululeko.feat_extract.feats_wavlm import Wavlm

        return Wavlm(model_name, config)

    elif "whisper" in model_lower:
        from nkululeko.feat_extract.feats_whisper import Whisper

        return Whisper(model_name, config)

    elif "ast" in model_lower:
        from nkululeko.feat_extract.feats_ast import Ast

        return Ast(model_name, config)

    elif "emotion2vec" in model_lower:
        from nkululeko.feat_extract.feats_emotion2vec import Emotion2vec

        return Emotion2vec(model_name, config)

    elif "opensmile" in model_lower or "gemaps" in model_lower or "compare" in model_lower:
        from nkululeko.feat_extract.feats_opensmile import Opensmileset

        return Opensmileset(model_name, config)

    elif "clap" in model_lower:
        from nkululeko.feat_extract.feats_clap import ClapSet

        return ClapSet(model_name, config)

    elif "spkrec" in model_lower or "xvect" in model_lower or "ecapa" in model_lower:
        from nkululeko.feat_extract.feats_spkrec import Spkrec

        return Spkrec(model_name, config)

    elif "trill" in model_lower:
        from nkululeko.feat_extract.feats_trill import TRILLset

        return TRILLset(model_name, config)

    elif "praat" in model_lower:
        from nkululeko.feat_extract.feats_praat import PraatSet

        return PraatSet(model_name, config)

    elif "audmodel" in model_lower:
        from nkululeko.feat_extract.feats_audmodel import AudmodelSet

        # Create audmodel extractor and load the model
        extractor = AudmodelSet(model_name, None, model_name)
        extractor._load_model()
        return extractor

    else:
        # Default to wav2vec2 if unknown
        print(
            f"WARNING: Unknown model {model_name}, defaulting to wav2vec2-large-robust-ft-swbd-300h"
        )
        from nkululeko.feat_extract.feats_wav2vec2 import Wav2vec2

        return Wav2vec2("wav2vec2-large-robust-ft-swbd-300h", config)


if __name__ == "__main__":
    main()
