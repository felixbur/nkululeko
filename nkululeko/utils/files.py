#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# copied from librosa.util.files.py

"""Utility functions for dealing with files"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, List, Optional, Set, Union

# add new function here
__all__ = [
    "find_files",
]


def find_files(
    directory: Union[str, os.PathLike[Any]],
    *,
    ext: Optional[Union[str, List[str]]] = None,
    recurse: bool = True,
    case_sensitive: bool = False,
    relative: bool = False,
    path_object: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[str]:
    """Get a sorted list of (audio) files in a directory or directory sub-tree.

    Examples
    --------
    >>> from nkululeko.utils import files
    >>> # Get all audio files in a directory sub-tree
    >>> files = files.find_files('~/Music')

    >>> # Look only within a specific directory, not the sub-tree
    >>> files = files.find_files('~/Music', recurse=False)

    >>> # Only look for mp3 files and return a list of pathlib.Path objects
    >>> files = files.find_files('~/Music', ext='mp3', path_object=True)

    >>> # Or just mp3 and ogg
    >>> files = files.find_files('~/Music', ext=['mp3', 'ogg'])

    >>> # Only get the first 10 files and relative paths
    >>> files = files.find_files('~/Music', limit=10, relative=True)

    >>> # Or last 10 files
    >>> files = files.find_files('~/Music', offset=-10)

    >>> # Avoid including search patterns in the path string
    >>> import glob
    >>> directory = '~/[202206] Music'
    >>> directory = glob.escape(directory)  # Escape the special characters
    >>> files = files.find_files(directory)

    Parameters
    ----------
    directory : str
        Path to look for files

    ext : str or list of str
        A file extension or list of file extensions to include in the search.

        Default: ``['aac', 'au', 'flac', 'm4a', 'mp3', 'ogg', 'wav']``

    recurse : boolean
        If ``True``, then all subfolders of ``directory`` will be searched.

        Otherwise, only ``directory`` will be searched.

    case_sensitive : boolean
        If ``False``, files matching upper-case version of
        extensions will be included.

    path_object : boolean
        If ``True``, then return a list of ``pathlib.Path`` objects.
        Otherwise, return a list of strings. Default: ``False``

    limit : int > 0 or None
        Return at most ``limit`` files. If ``None``, all files are returned.

    offset : int
        Return files starting at ``offset`` within the list.

        Use negative values to offset from the end of the list.

    Returns
    -------
    files : list of str
        The list of audio files.
    """
    if ext is None:
        ext = ["aac", "au", "flac", "m4a", "mp3", "ogg", "wav"]

    elif isinstance(ext, str):
        ext = [ext]

    # Cast into a set
    ext = set(ext)

    # Generate upper-case versions
    if not case_sensitive:
        # Force to lower-case
        ext = {e.lower() for e in ext}
        # Add in upper-case versions
        ext |= {e.upper() for e in ext}

    fileset = set()

    if recurse:
        for walk in os.walk(directory):  # type: ignore
            fileset |= __get_files(walk[0], ext)
    else:
        fileset = __get_files(directory, ext)

    files = list(fileset)
    files.sort()
    files = files[offset:]
    if limit is not None:
        files = files[:limit]

    if relative:
        files = [os.path.relpath(f) for f in files]

    if path_object:
        files = [Path(f) for f in files]

    return files


def __get_files(dir_name: Union[str, os.PathLike[Any]], extensions: Set[str]):
    """Get a list of files in a single directory"""
    # Expand out the directory
    dir_name = os.path.abspath(os.path.expanduser(dir_name))

    myfiles = set()

    for sub_ext in extensions:
        globstr = os.path.join(dir_name, "*" + os.path.extsep + sub_ext)
        myfiles |= set(glob.glob(globstr))

    return myfiles
