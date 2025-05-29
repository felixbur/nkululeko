#!/usr/bin/env python3

# unzip.py - Extracts all files from the specified zip files (in case unzip is not installed).
# usage: python unzip.py file1.zip file2.zip ...
# source: https://askubuntu.com/questions/86849/how-to-unzip-a-zip-file-from-the-terminal

# mimics the unzip command
# # To extract an archive to a specific directory:
# unzip <archive>.zip -d <directory>

import argparse
from zipfile import ZipFile

# Set up argument parser
parser = argparse.ArgumentParser(description="Extract files from zip archives.")
parser.add_argument("zip_files", nargs="+", help="List of zip files to extract.")
parser.add_argument(
    "-d",
    "--directory",
    default=".",
    help="Directory to extract files to (default: current directory).",
)
args = parser.parse_args()

# Extract arguments
directory = args.directory
zip_files = args.zip_files

# Process each zip file
for zip_file in zip_files:
    try:
        with ZipFile(zip_file) as zf:
            zf.extractall(path=directory)
        print(f"Extracting {zip_file} to {directory}")
    except FileNotFoundError:
        print(f"Error: File {zip_file} not found")
    except Exception as e:
        print(f"Error extracting {zip_file}: {e}")
