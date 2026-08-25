"""Tests for nkululeko/utils/files.py — find_files and find_files_by_name."""

import ntpath
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nkululeko.utils.files import find_files, find_files_by_name, mirror_relpath


@pytest.fixture
def audio_tree(tmp_path):
    """Create a small directory tree with audio and non-audio files."""
    # Root level
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "b.mp3").write_bytes(b"")
    (tmp_path / "readme.txt").write_bytes(b"")
    # Sub-directory
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.wav").write_bytes(b"")
    (sub / "d.flac").write_bytes(b"")
    (sub / "notes.md").write_bytes(b"")
    # Nested sub-directory
    nested = sub / "nested"
    nested.mkdir()
    (nested / "e.wav").write_bytes(b"")
    return tmp_path


class TestFindFiles:
    def test_finds_all_audio_recursively(self, audio_tree):
        result = find_files(audio_tree)
        names = {os.path.basename(f) for f in result}
        assert names == {"a.wav", "b.mp3", "c.wav", "d.flac", "e.wav"}

    def test_non_audio_files_excluded(self, audio_tree):
        result = find_files(audio_tree)
        names = {os.path.basename(f) for f in result}
        assert "readme.txt" not in names
        assert "notes.md" not in names

    def test_no_recurse_finds_only_root(self, audio_tree):
        result = find_files(audio_tree, recurse=False)
        names = {os.path.basename(f) for f in result}
        assert names == {"a.wav", "b.mp3"}

    def test_single_extension_filter(self, audio_tree):
        result = find_files(audio_tree, ext="wav")
        names = {os.path.basename(f) for f in result}
        assert names == {"a.wav", "c.wav", "e.wav"}

    def test_multiple_extension_filter(self, audio_tree):
        result = find_files(audio_tree, ext=["wav", "flac"])
        names = {os.path.basename(f) for f in result}
        assert names == {"a.wav", "c.wav", "d.flac", "e.wav"}

    def test_result_is_sorted(self, audio_tree):
        result = find_files(audio_tree)
        assert result == sorted(result)

    def test_limit_restricts_count(self, audio_tree):
        result = find_files(audio_tree, limit=2)
        assert len(result) == 2

    def test_positive_offset_skips_files(self, audio_tree):
        all_files = find_files(audio_tree)
        result = find_files(audio_tree, offset=2)
        assert result == all_files[2:]

    def test_negative_offset_gives_last_n(self, audio_tree):
        all_files = find_files(audio_tree)
        result = find_files(audio_tree, offset=-2)
        assert result == all_files[-2:]

    def test_relative_paths(self, audio_tree, monkeypatch):
        monkeypatch.chdir(audio_tree)
        result = find_files(audio_tree, recurse=False, relative=True)
        assert all(not os.path.isabs(f) for f in result)

    def test_path_object_returns_path_instances(self, audio_tree):
        result = find_files(audio_tree, recurse=False, path_object=True)
        assert all(isinstance(f, Path) for f in result)

    def test_case_insensitive_extension(self, tmp_path):
        (tmp_path / "upper.WAV").write_bytes(b"")
        result = find_files(tmp_path, ext="wav", case_sensitive=False)
        names = {os.path.basename(f) for f in result}
        assert "upper.WAV" in names

    def test_empty_directory_returns_empty(self, tmp_path):
        result = find_files(tmp_path)
        assert result == []


class TestFindFilesByName:
    def test_finds_files_matching_pattern(self, audio_tree):
        result = find_files_by_name(audio_tree, "notes")
        names = {os.path.basename(f) for f in result}
        assert "notes.md" in names

    def test_case_insensitive_by_default(self, tmp_path):
        (tmp_path / "ExpResult.txt").write_bytes(b"")
        result = find_files_by_name(tmp_path, "expresult", case_sensitive=False)
        names = {os.path.basename(f) for f in result}
        assert "ExpResult.txt" in names

    def test_case_sensitive_excludes_wrong_case(self, tmp_path):
        (tmp_path / "ExpResult.txt").write_bytes(b"")
        result = find_files_by_name(tmp_path, "expresult", case_sensitive=True)
        names = {os.path.basename(f) for f in result}
        assert "ExpResult.txt" not in names

    def test_no_recurse_searches_only_root(self, audio_tree):
        result = find_files_by_name(audio_tree, "notes", recurse=False)
        assert result == []

    def test_limit_restricts_count(self, tmp_path):
        for i in range(5):
            (tmp_path / f"test_{i}.txt").write_bytes(b"")
        result = find_files_by_name(tmp_path, "test", limit=3)
        assert len(result) == 3

    def test_positive_offset(self, tmp_path):
        for i in range(4):
            (tmp_path / f"file_{i}.txt").write_bytes(b"")
        all_results = find_files_by_name(tmp_path, "file")
        offset_results = find_files_by_name(tmp_path, "file", offset=2)
        assert offset_results == all_results[2:]

    def test_relative_paths(self, audio_tree):
        result = find_files_by_name(audio_tree, "readme", recurse=False, relative=True)
        assert all(not os.path.isabs(f) for f in result)

    def test_path_object(self, tmp_path):
        (tmp_path / "myconfig.ini").write_bytes(b"")
        result = find_files_by_name(tmp_path, "myconfig", path_object=True)
        assert all(isinstance(f, Path) for f in result)

    def test_no_match_returns_empty(self, audio_tree):
        result = find_files_by_name(audio_tree, "zzz_not_there")
        assert result == []

    def test_result_is_sorted(self, tmp_path):
        for name in ["z_match.txt", "a_match.txt", "m_match.txt"]:
            (tmp_path / name).write_bytes(b"")
        result = find_files_by_name(tmp_path, "match")
        assert result == sorted(result)


class TestMirrorRelpath:
    """mirror_relpath() lets augmenters mirror a source file's directory
    structure under their own cache dir without colliding across datasets.
    Real behavior is exercised on this (POSIX) host; Windows drive-letter
    handling is exercised by swapping in ntpath's semantics, since that
    can't otherwise be triggered on a POSIX CI runner."""

    def test_posix_absolute_path_becomes_relative(self, tmp_path):
        f = tmp_path / "audio" / "f0.wav"
        rel = mirror_relpath(str(f))
        assert not os.path.isabs(rel)
        # joining back under an arbitrary root must land inside that root
        root = "/some/cache/root"
        joined = os.path.join(root, rel)
        assert joined.startswith(root + os.sep)

    def test_posix_two_dirs_same_basename_stay_distinct(self, tmp_path):
        path_a = str(tmp_path / "dataset_a" / "wav" / "f0.wav")
        path_b = str(tmp_path / "dataset_b" / "wav" / "f0.wav")
        assert mirror_relpath(path_a) != mirror_relpath(path_b)

    def _windows_mirror_relpath(self, path):
        """Call mirror_relpath with Windows path semantics substituted in,
        regardless of the host OS running the test."""
        with patch("nkululeko.utils.files.os.path", ntpath), patch(
            "nkululeko.utils.files.os.sep", ntpath.sep
        ):
            return mirror_relpath(path)

    def test_windows_drive_letter_is_stripped_not_left_absolute(self):
        rel = self._windows_mirror_relpath(r"C:\Users\foo\audio\f0.wav")
        assert not ntpath.isabs(rel)
        assert ":" not in rel

    def test_windows_join_stays_rooted_under_cache_dir(self):
        """The original bug: os.path.join(root, rel) must not silently
        discard root because rel still looks like a drive-absolute path."""
        rel = self._windows_mirror_relpath(r"C:\Users\foo\audio\f0.wav")
        root = r"C:\cache\silero"
        joined = ntpath.join(root, rel)
        assert joined.startswith(root + ntpath.sep)

    def test_windows_different_drives_stay_distinct(self):
        rel_c = self._windows_mirror_relpath(r"C:\audio\f0.wav")
        rel_d = self._windows_mirror_relpath(r"D:\audio\f0.wav")
        assert rel_c != rel_d
