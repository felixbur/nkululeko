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
                    audmodel (requires audmodel.id in config), agender, opensmile, clap, spkrec, trill, praat,
                    squim (PESQ/SDR/STOI), mos, snr, etc.
"""

import argparse
import configparser
import os
import soundfile as sf

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
    parser.add_argument("--config", help="The base configuration")
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
    )
    parser.add_argument(
        "--folder",
        help=("A name of a folder where the files within the list are in."),
        nargs="?",
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
            " audmodel, agender, opensmile, clap, spkrec, trill, praat, squim, mos, snr)"
        ),
        nargs="?",
        default="wav2vec2-large-robust-ft-swbd-300h",
    )
    args = parser.parse_args()

    # Handle configuration file
    if args.config is not None:
        # User explicitly provided a config file
        config_file = args.config
        if not os.path.isfile(config_file):
            print("=" * 80)
            print(f"ERROR: Configuration file not found: {config_file}")
            print("\nPlease create a config file with the following structure:")
            print("  [EXP]")
            print("  name = feature_demo")
            print("\n  [FEATS]")
            print("  audmodel.id = <model_id>  # Required for audmodel")
            print("  wav2vec2.layer = <layer>  # Optional for wav2vec2")
            print("\n  [MODEL]")
            print("  device = cuda  # or cpu")
            print("=" * 80)
            exit(1)
        # load configuration
        config = configparser.ConfigParser()
        config.read(config_file)
        print(f"Using configuration from: {config_file}")
    else:
        # No config provided, use defaults
        print("No config file specified, using default settings")
        config = configparser.ConfigParser()
        config["EXP"] = {"name": "feature_demo"}
        config["DATA"] = {"databases": ["none"]}
        config["FEATS"] = {"type": [args.model]}
        config["MODEL"] = {
            "type": "svm",
            "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        }
        config["FEATS"]["needs_feature_extraction"] = "True"
        config["FEATS"]["no_reuse"] = "True"

    # Initialize global config
    glob_conf.config = config

    # Collect files to process - do this BEFORE initializing the heavy models
    files = []
    if args.mic:
        # Record from microphone first before loading models
        print("\n" + "=" * 80, flush=True)
        print("MICROPHONE RECORDING", flush=True)
        print("=" * 80, flush=True)

        duration = 5  # seconds
        sample_rate = 16000  # Hz

        print("Recording NOW for 5 seconds - speak clearly!", flush=True)
        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()  # Wait until recording is finished
        print("Recording finished!", flush=True)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_filename = temp_file.name
        temp_file.close()

        # Write the recording to the temp file
        sf.write(temp_filename, recording, sample_rate)
        print(f"Audio saved to temporary file: {temp_filename}")

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
        print("=" * 80)
        print("ERROR: No input option specified!")
        print("=" * 80)
        print("\nYou must provide one of the following input options:")
        print("  --file <path>      Process a single audio file")
        print("  --list <path>      Process files listed in a text file")
        print("  --folder <path>    Process all audio files in a folder")
        print("  --mic              Record from microphone for 5 seconds")
        print("\nExample usage:")
        print("  python -m nkululeko.feature_demo --mic --model agender")
        print("  python -m nkululeko.feature_demo --file audio.wav --model squim")
        print("  python -m nkululeko.feature_demo --folder ./audio --model mos")
        print("=" * 80)
        exit(1)

    print(f"\nProcessing {len(files)} file(s)...")

    # Initialize Util AFTER collecting files
    module = "feature_demo"
    util = Util(module)
    util.debug(f"running feature extraction demo, nkululeko version {VERSION}")
    util.debug(f"using model: {args.model}")

    # Initialize feature extractor based on model type
    print(f"\nInitializing {args.model} model...")
    print("(This may take a while on first run - downloading models...)")
    feature_extractor = get_feature_extractor(args.model, config)
    print("Model initialized successfully!")

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
            if isinstance(features, (int, float)):
                # Single scalar value (e.g., SNR, MOS)
                features = np.array([features])
            elif isinstance(features, tuple):
                # Tuple of values (e.g., SQUIM returns (stoi, pesq, sdr))
                features = np.array(features)
            elif isinstance(features, np.ndarray):
                if features.ndim > 1:
                    # Flatten if multi-dimensional
                    features = features.flatten()
            elif isinstance(features, pd.Series):
                features = features.values
            elif isinstance(features, pd.DataFrame):
                features = features.values.flatten()
            else:
                # Try to convert to numpy array
                features = np.array(features).flatten()

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
            outfile = (
                args.outfile if args.outfile.endswith(".csv") else args.outfile + ".csv"
            )
            df.to_csv(outfile)
            print(f"\nFeatures saved to {outfile}")

        # Print summary to stdout
        print("\n" + "=" * 80)
        print("FEATURE EXTRACTION SUMMARY")
        print("=" * 80)
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
            print("Cleaned up temporary recording file")
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

        print(f"  Loading Wav2Vec2 model: {model_name}...")
        extractor = Wav2vec2(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif "hubert" in model_lower:
        from nkululeko.feat_extract.feats_hubert import Hubert

        extractor = Hubert(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif "wavlm" in model_lower:
        from nkululeko.feat_extract.feats_wavlm import Wavlm

        extractor = Wavlm(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif "whisper" in model_lower:
        from nkululeko.feat_extract.feats_whisper import Whisper

        extractor = Whisper(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif "ast" in model_lower:
        from nkululeko.feat_extract.feats_ast import Ast

        extractor = Ast(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif "emotion2vec" in model_lower:
        from nkululeko.feat_extract.feats_emotion2vec import Emotion2vec

        extractor = Emotion2vec(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif (
        "opensmile" in model_lower
        or "gemaps" in model_lower
        or "compare" in model_lower
    ):
        from nkululeko.feat_extract.feats_opensmile import Opensmileset

        extractor = Opensmileset(model_name, None, model_name)
        # OpenSMILE doesn't need explicit init_model
        return extractor

    elif "clap" in model_lower:
        from nkululeko.feat_extract.feats_clap import ClapSet

        extractor = ClapSet(model_name, None, model_name)
        extractor._load_model()
        return extractor

    elif "spkrec" in model_lower or "xvect" in model_lower or "ecapa" in model_lower:
        from nkululeko.feat_extract.feats_spkrec import Spkrec

        extractor = Spkrec(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif "trill" in model_lower:
        from nkululeko.feat_extract.feats_trill import TRILLset

        extractor = TRILLset(model_name, None, model_name)
        extractor._load_model()
        return extractor

    elif "praat" in model_lower:
        from nkululeko.feat_extract.feats_praat import PraatSet

        extractor = PraatSet(model_name, None, model_name)
        # Praat doesn't need explicit init
        return extractor

    elif "audmodel" in model_lower:
        from nkululeko.feat_extract.feats_audmodel import AudmodelSet

        # Create audmodel extractor and load the model
        extractor = AudmodelSet(model_name, None, model_name)
        extractor._load_model()
        return extractor

    elif "agender" in model_lower:
        from nkululeko.feat_extract.feats_agender_agender import Agender_agenderSet

        # Create agender extractor and load the model
        print(
            "  Loading agender_agender model (this may download the model on first use)..."
        )
        extractor = Agender_agenderSet(model_name, None, model_name)
        extractor._load_model()
        return extractor

    elif "squim" in model_lower:
        from nkululeko.feat_extract.feats_squim import SquimSet

        # SQUIM extracts PESQ, SDR (SI-SDR), and STOI metrics
        print("  Loading SQUIM model for audio quality metrics (PESQ, SDR, STOI)...")
        extractor = SquimSet(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif "pesq" in model_lower:
        from nkululeko.feat_extract.feats_squim import SquimSet

        # PESQ (Perceptual Evaluation of Speech Quality) via SQUIM
        print("  Loading SQUIM model to extract PESQ features...")
        extractor = SquimSet(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif "sdr" in model_lower:
        from nkululeko.feat_extract.feats_squim import SquimSet

        # SDR (Signal-to-Distortion Ratio) via SQUIM
        print("  Loading SQUIM model to extract SDR features...")
        extractor = SquimSet(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif "mos" in model_lower:
        from nkululeko.feat_extract.feats_mos import MosSet

        # MOS extracts Mean Opinion Score for subjective quality
        print("  Loading MOS model for subjective quality assessment...")
        extractor = MosSet(model_name, None, model_name)
        extractor.init_model()
        return extractor

    elif "snr" in model_lower:
        from nkululeko.feat_extract.feats_snr import SnrSet

        # SNR extracts Signal-to-Noise Ratio
        print("  Loading SNR estimator...")
        extractor = SnrSet(model_name, None, model_name)
        # SNR doesn't need explicit model initialization
        return extractor

    else:
        # Default to wav2vec2 if unknown
        print(f"WARNING: Unknown model {model_name}")
        exit(1)


if __name__ == "__main__":
    main()
