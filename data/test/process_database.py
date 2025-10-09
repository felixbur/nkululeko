import diffsptk
import pandas as pd
import numpy as np
from datetime import timedelta


def generate_frame_labels(
    audio_path, frame_period=80, f_min=80, f_max=180, voicing_threshold=0.4
):
    """
    Generate frame-level voiced/unvoiced labels for a given audio file.

    Args:
        audio_path (str): Path to the audio file.
        frame_period (int): Frame shift period in samples.
        f_min (float): Minimum frequency for pitch detection.
        f_max (float): Maximum frequency for pitch detection.
        voicing_threshold (float): Voicing threshold for pitch detection.

    Returns:
        np.ndarray: Frame-level labels (0 for unvoiced, 1 for voiced).
    """
    # Read audio file
    x, sr = diffsptk.read(audio_path)

    # Pitch extraction for voicing detection
    pitch = diffsptk.Pitch(
        frame_period=frame_period,
        sample_rate=sr,
        f_min=f_min,
        f_max=f_max,
        voicing_threshold=voicing_threshold,
        out_format="f0",
    )
    p = pitch(x)
    labels = np.asarray(p != 0).astype(int)  # 1 for voiced, 0 for unvoiced

    return labels


if __name__ == "__main__":
    # List of audio files
    files = [
        "03a01Fa.wav",
        "03a01Nc.wav",
        "03a01Wa.wav",
        "03a02Ta.wav",
        "03a04Ad.wav",
        "03a04Lc.wav",
    ]

    # Prepare metadata for all frames
    metadata = []

    for file in files:
        # Read audio to get sample rate
        from diffsptk import read

        _, sr = read(f"audio/{file}")

        # Generate frame-level labels for the current file
        labels = generate_frame_labels(f"audio/{file}")

        # Add each frame's label to the metadata with start and end times
        for frame_idx, label in enumerate(labels):
            # Calculate start and end times in seconds based on frame_idx and frame_period
            # frame_period is in samples, so convert to seconds using sample rate
            start_seconds = (frame_idx * 80) / sr  # 80 is frame_period in samples
            end_seconds = ((frame_idx + 1) * 80) / sr

            # Convert to timedelta format: "0 days HH:MM:SS.microseconds"
            start_td = timedelta(seconds=start_seconds)
            end_td = timedelta(seconds=end_seconds)

            # Format with "0 days" prefix
            start_time = f"0 days {str(start_td)}"
            end_time = f"0 days {str(end_td)}"

            metadata.append(
                {
                    "file": f"audio/{file}",
                    "start": start_time,
                    "end": end_time,
                    "frame_idx": frame_idx,
                    "frame_period": 80,
                    "voiced": label,
                }
            )

    # Save the metadata to a CSV file
    df = pd.DataFrame(metadata)
    # allocate "03a04Lc.wav" to test set, others to train set
    df["set"] = np.where(df["file"] == "audio/03a04Lc.wav", "test", "train")
    # save train set to voiced_train.csv and test set to voiced_test.csv
    df[df["set"] == "train"].to_csv("voiced_train.csv", index=False)
    print(f"Train set shape: {df[df['set'] == 'train'].shape} to voiced_train.csv")
    df[df["set"] == "test"].to_csv("voiced_test.csv", index=False)
    print(f"Test set shape: {df[df['set'] == 'test'].shape} to voiced_test.csv")
