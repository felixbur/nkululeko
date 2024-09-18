"""
Segment audio files in a given directory into smaller chunks with a specified length and overlap. This requires the PyDub library to be installed.

Args:
    input_dir (str): The directory containing the audio files to be segmented.
    output_dir (str): The directory where the segmented audio files will be saved.
    segment_length (int): The length of each audio segment in milliseconds.
    overlap (int): The overlap between adjacent audio segments in milliseconds.

This function will recursively search the input directory for all .wav audio files, and then segment each file into smaller chunks with the specified length and overlap. The segmented audio files will be saved in the output directory, preserving the relative directory structure from the input directory.
"""

import argparse
import glob
from pathlib import Path

from pydub import AudioSegment


# list audio files given a directory
def segment_audio(input_dir, output_dir, segment_length, overlap):
    # check if input dir exist
    if not Path(input_dir).exists():
        print(f"Directory {input_dir} does not exist.")
        return

    # check if output dir exist, create if not
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    audio_files = glob.glob(str(Path(input_dir) / "**" / "*.wav"), recursive=True)

    for audio_file in audio_files:
        # segment into 2 seconds with 1 second overlap (default values)
        audio = AudioSegment.from_file(audio_file)

        segments = []

        for i in range(0, len(audio), segment_length - overlap):
            segment = audio[i : i + segment_length]
            segments.append(segment)

        # Path(output_dir).mkdir(exist_ok=True)
        for i, segment in enumerate(segments):
            # get relative path from input_dir
            relative_path = Path(audio_file).relative_to(input_dir)
            # make output directory if not exist
            output_subdir = Path(output_dir) / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            segment.export(
                str(output_subdir / f"{Path(audio_file).stem}_{i}.wav"),
                format="wav",
            )

    print("DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./")
    # add argument for output_dir
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./segmented_data/",
    )
    parser.add_argument(
        "--segment_length",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=1000,
    )
    args = parser.parse_args()
    segment_audio(
        args.input_dir,
        args.output_dir,
        segment_length=args.segment_length,
        overlap=args.overlap,
    )
