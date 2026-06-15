"""creates splits (train and test) for the database as well as applies spectral instability to identify speech tasks within an audio file"""

import audiofile as af
import scipy.spatial as sp
import librosa
import torch
from audformat import segmented_index

import numpy as np
import pandas as pd

import numpy as np

from splitutils import binning, optimize_traintest_split

from pydub import AudioSegment

SAMPLING_RATE = 16000

vad_model, vad_utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False,
)


def stereo_to_mono(input_file, output_file):
    sound = AudioSegment.from_file(input_file, format="wav")

    sound = sound.set_channels(1)

    sound.export(output_file, format="wav")


def create_train_test_split(df: pd.DataFrame) -> pd.DataFrame:

    print(f"Performing train-test split...")
    seed = 42
    np.random.seed(seed)
    test_size = 0.2
    k = 30

    label = df["grbas_score"].to_numpy()

    label = binning(label, nbins=3)

    aesthenia_score = binning(df["grbas_aesthenia_score"].to_numpy(), nbins=3)
    breathiness_score = binning(df["grbas_breathiness_score"].to_numpy(), nbins=3)
    roughness_score = binning(df["grbas_roughness_score"].to_numpy(), nbins=3)
    strain_score = binning(df["grbas_strain_score"].to_numpy(), nbins=3)

    speaker = df["speaker"].to_numpy()
    stratif_vars = {
        "diagnosis": label,
        "strat_var1": aesthenia_score,
        "strat_var2": breathiness_score,
        "strat_var3": roughness_score,
        "strat_var4": strain_score,
    }
    # weights for all stratify_on variables and
    # and for test proportion match.
    weight = {
        "diagnosis": 2,
        "strat_var1": 1,
        "strat_var2": 1,
        "strat_var3": 1,
        "strat_var4": 1,
        "size_diff": 1,
    }
    # find optimal test indices TEST_I in DF
    train_i, test_i, info = optimize_traintest_split(
        X=df,
        y=label,
        split_on=speaker,
        stratify_on=stratif_vars,
        weight=weight,
        test_size=test_size,
        k=k,
        seed=seed,
    )

    df["split"] = "unknown"

    df.loc[test_i, "split"] = "test"
    df.loc[train_i, "split"] = "train"

    return df

def get_segmentation_simple(file):
    (
        get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks,
    ) = vad_utils
    SAMPLING_RATE = 16000
    wav = read_audio(file, sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(
        wav, vad_model, sampling_rate=SAMPLING_RATE
    )
    files, starts, ends = [], [], []
    for entry in speech_timestamps:
        start = float(entry["start"] / SAMPLING_RATE)
        end = float(entry["end"] / SAMPLING_RATE)
        files.append(file)
        starts.append(start)
        ends.append(end)
    seg_index = segmented_index(files, starts, ends)
    return seg_index


def vad_segmentation(df: pd.DataFrame, strip_prefix: str = "") -> pd.DataFrame:
    """obtain VAD segments from data"""

    df.reset_index(inplace=True)
    df_vad = pd.DataFrame()

    for f in df["file"].unique():
        y, fs = af.read(f)
        if af.channels(f) > 1:
            print(f"Stereo file detected: {f}, converting to mono")
            # for some reason there were some files with two channels, quick manual inspection did not show two channels though
            stereo_to_mono(f, f)
            y, fs = af.read(f)  # Reload the now-mono audio
        df_seg = get_segmentation_simple(f)
        df_seg = df_seg.to_frame(index=False)

        # no vad detected?
        if df_seg.shape[0] == 0:
            dur = float(len(y) / fs)
            s = pd.to_timedelta(np.array([0.0]), unit="s")
            e = pd.to_timedelta(np.array([dur]), unit="s")
            df_seg = pd.DataFrame({"start": s, "end": e})

        df_seg["file"] = np.array([f] * df_seg.shape[0], dtype=str)

        # grouping: copy metadata but preserve VAD-derived file/start/end
        df_fi = df.loc[(df["file"] == f)]
        skip = {"file", "start", "end"}
        for c in df_fi.columns:
            if c not in skip:
                df_seg[c] = [df_fi[c].iloc[0]] * df_seg.shape[0]

        df_vad = pd.concat((df_vad, df_seg), ignore_index=True)

    df_vad = get_speech_task(df_vad)

    if strip_prefix:
        df_vad["file"] = df_vad["file"].str.replace(strip_prefix.rstrip("/") + "/", "", n=1, regex=False)

    return df_vad


def get_speech_task(df):
    """get speech task based on spectral instability"""

    df = df.copy()
    df = dt2sec(df)
    threshold = 175.393

    # Get unique filenames
    unique_files = df["file"].unique()

    speech_task = []
    for filename in unique_files:
        file_df = df[df["file"] == filename]
        for _, row in file_df.iterrows():
            start = row["start"]
            end = row["end"]

            si = spectral_instability(filename, start, end)
            if si["median"] > threshold:
                speech_task.append("read_speech")
            else:
                speech_task.append("sustained_utterance")

    df["speech_task"] = speech_task

    # Convert "start" and "end" back to timedelta
    df["start"] = pd.to_timedelta(df["start"], unit="s")
    df["end"] = pd.to_timedelta(df["end"], unit="s")

    return df


def spectral_instability(
    filename: str,
    start: float = None,
    end: float = None,
    n_mfcc: int = 12,
    n_mels: int = 24,
    n_fft: int = 512,
    hop_length: int = 256,
    distance: str = "euclidean",
    pairing: str = "center",
    lifter: float = 24.0,
) -> dict:
    """Compute spectral instability using librosa.

    Args:
        filename: path to audio file
        start: segment start in seconds (None = file start)
        end: segment end in seconds (None = file end)
        n_mfcc: number of MFCC coefficients
        n_mels: number of mel filter banks
        n_fft: FFT window size in samples
        hop_length: hop size in samples
        distance: "euclidean" or "canberra"
        pairing: "center", "first", or "adjacent"
        lifter: liftering coefficient (0 = no liftering)

    Returns:
        dict with keys "mean", "median", "std"
    """
    duration = None if (start is None or end is None) else end - start
    y, sr = librosa.load(filename, sr=None, offset=start or 0.0, duration=duration, mono=True)

    x = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, center=False
    )

    # apply liftering: w[n] = 1 + (lifter/2) * sin(pi*n/lifter)
    if lifter > 0:
        n = np.arange(n_mfcc, dtype=float)
        lift = 1.0 + (lifter / 2.0) * np.sin(np.pi * n / lifter)
        x = lift[:, np.newaxis] * x

    if x.ndim < 2 or x.shape[1] < 2:
        return {"mean": np.nan, "median": np.nan, "std": np.nan}

    # framewise mean subtraction
    m = np.mean(x, axis=0)
    x = x - np.tile(m, (x.shape[0], 1))

    # pairwise distances
    dist = []
    if pairing == "first":
        j = 0
    elif pairing == "center":
        j = int(x.shape[1] / 2)

    for i in range(x.shape[1]):
        if pairing in ["center", "first"] and i == j:
            continue
        elif pairing == "adjacent":
            j = i - 1
            if j < 0:
                continue
        if distance == "euclidean":
            d = np.linalg.norm(x[:, j] - x[:, i])
        else:  # canberra
            diff = np.abs(x[:, j] - x[:, i])
            denom = np.abs(x[:, j]) + np.abs(x[:, i])
            d = float(np.nansum(diff / denom))
        dist.append(d)

    dist = np.array(dist)
    return {"mean": float(np.mean(dist)), "median": float(np.median(dist)), "std": float(np.std(dist))}


def dt2sec(df, reset_index=True):
    """converts start and end index values from datetime
    to seconds.
    Args:
    df: pd.DataFrame in unified format
    reset_index: (boolean) if True, "file, start, end" will be
        columns. If False, they will be index
    """

    if "start" not in df.columns:
        df.reset_index(inplace=True)

    df["start"] = df["start"].dt.total_seconds()
    df["end"] = df["end"].dt.total_seconds()

    if not reset_index:
        df.set_index(["file", "start", "end"], inplace=True)

    return df
