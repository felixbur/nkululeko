# pickle_integrity.py - SHA256 checksum utilities for pickle file integrity
"""Provides functions to save and verify SHA256 checksums for pickle files.

Pickle deserialization can execute arbitrary Python code. These utilities
store a SHA256 checksum alongside each pickle file and verify it before
loading, protecting against tampered files on shared filesystems.
"""

import hashlib
import logging
import os

logger = logging.getLogger(__name__)


def checksum_path(pickle_path: str) -> str:
    """Return the path to the checksum file for a given pickle file."""
    return pickle_path + ".sha256"


_checksum_path = checksum_path


def save_checksum(pickle_path: str) -> None:
    """Compute and store a SHA256 checksum for a pickle file.

    Args:
        pickle_path: Path to the pickle file.
    """
    sha256 = hashlib.sha256()
    with open(pickle_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    checksum = sha256.hexdigest()
    with open(_checksum_path(pickle_path), "w") as f:
        f.write(checksum + "\n")
    logger.debug(f"Saved checksum for {pickle_path}")


def verify_checksum(pickle_path: str) -> None:
    """Verify the SHA256 checksum of a pickle file before loading.

    If no checksum file exists (e.g. legacy files created before this
    feature), a warning is logged but loading proceeds. This ensures
    backward compatibility.

    Args:
        pickle_path: Path to the pickle file.

    Raises:
        ValueError: If the checksum file exists but does not match.
    """
    checksum_file = _checksum_path(pickle_path)
    if not os.path.isfile(checksum_file):
        logger.warning(
            f"No checksum file found for {pickle_path}. "
            "Pickle files should only be loaded from trusted sources."
        )
        return

    with open(checksum_file, "r") as f:
        expected = f.read().strip()

    sha256 = hashlib.sha256()
    with open(pickle_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()

    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {pickle_path}: "
            f"expected {expected}, got {actual}. "
            "The file may have been tampered with."
        )
    logger.debug(f"Checksum verified for {pickle_path}")
