#!/usr/bin/env python3

# unzip.py - Extracts all files from the specified zip files (in case unzip is not installed).
# usage: python unzip.py file1.zip file2.zip ...
# source: https://askubuntu.com/questions/86849/how-to-unzip-a-zip-file-from-the-terminal

# mimics the unzip command
# # To extract an archive to a specific directory:
# unzip <archive>.zip -d <directory>

import sys
from zipfile import ZipFile

# if no zip files are provided, print usage and exit
if len(sys.argv) < 2:
    print("Usage: python unzip.py file1.zip file2.zip -d <directory>")
    sys.exit(1)

# Check if -d flag is provided and determine directory and zip files
if len(sys.argv) >= 3 and sys.argv[-2] == "-d":
    directory = sys.argv[-1]
    zip_files = sys.argv[1:-2]
else:
    import os

    directory = os.getcwd()
    zip_files = sys.argv[1:]

# Process each zip file
for zip_file in zip_files:
    with ZipFile(zip_file) as zf:
        zf.extractall(path=directory)
    print(f"Extracting {zip_file} to {directory}.")
