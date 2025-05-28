#!/usr/bin/env python3

# unzip.py - Extracts all files from the specified zip files (in case unzip is not installed).
# usage: python unzip.py file1.zip file2.zip ...
# source: https://askubuntu.com/questions/86849/how-to-unzip-a-zip-file-from-the-terminal

import sys
from zipfile import PyZipFile

for zip_file in sys.argv[1:]:
    with PyZipFile(zip_file) as pzf:
        pzf.extractall()
