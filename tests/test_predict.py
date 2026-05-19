"""Tests for the unified nkululeko.predict module.

Covers:

- pure helpers (`_first_extractor`, `_flatten_features`, `_build_segmented_df`)
- the AUTOPREDICT_TARGETS table
- the argparse CLI (`_build_parser`)
- config loading (`_load_config`)
- input-mode handlers (`_run_files`, `_run_list`, `_run_folder`) with the
  prediction backend mocked
- dispatch logic (`_predict_df`) between model / autopredict / feature paths
- the backwards-compatible `do_test` entry point
- `main()` smoke behaviour (`--help`, missing input)
"""

import argparse
import configparser
import os
import sys
from unittest.mock import MagicMock, patch

import audformat
import numpy as np
import pandas as pd
import pytest
import soundfile as sf


def _write_silent_wav(path, samples=1600, sr=16000):
    """Write a tiny valid WAV file used by tests that need a real audio file
    on disk (e.g. anything that exercises `_resolve_nat_ends`)."""
    sf.write(str(path), np.zeros(samples, dtype="float32"), sr)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestFirstExtractor:
    """`_first_extractor` normalizes the various FEATS.type encodings."""

    def test_list_value(self):
        from nkululeko.predict import _first_extractor

        assert _first_extractor(["wav2vec2", "opensmile"]) == "wav2vec2"

    def test_empty_list_value(self):
        from nkululeko.predict import _first_extractor

        assert _first_extractor([]) is None

    def test_python_literal_list_string(self):
        from nkululeko.predict import _first_extractor

        assert _first_extractor("['wav2vec2', 'opensmile']") == "wav2vec2"

    def test_plain_string(self):
        from nkululeko.predict import _first_extractor

        assert _first_extractor("audmodel") == "audmodel"

    def test_comma_separated(self):
        from nkululeko.predict import _first_extractor

        assert _first_extractor("wav2vec2,opensmile") == "wav2vec2"

    def test_quoted_single_string(self):
        from nkululeko.predict import _first_extractor

        # ast.literal_eval interprets a single quoted string literal.
        assert _first_extractor("'audmodel'") == "audmodel"


class TestFlattenFeatures:
    """`_flatten_features` produces a 1-D ndarray for every supported input."""

    def test_scalar(self):
        from nkululeko.predict import _flatten_features

        np.testing.assert_array_equal(_flatten_features(3.14), np.array([3.14]))

    def test_int_scalar(self):
        from nkululeko.predict import _flatten_features

        np.testing.assert_array_equal(_flatten_features(2), np.array([2]))

    def test_tuple(self):
        from nkululeko.predict import _flatten_features

        np.testing.assert_array_equal(
            _flatten_features((1.0, 2.0, 3.0)), np.array([1.0, 2.0, 3.0])
        )

    def test_1d_array(self):
        from nkululeko.predict import _flatten_features

        arr = np.array([1, 2, 3])
        np.testing.assert_array_equal(_flatten_features(arr), arr)

    def test_2d_array_flattens(self):
        from nkululeko.predict import _flatten_features

        out = _flatten_features(np.array([[1, 2], [3, 4]]))
        assert out.shape == (4,)
        np.testing.assert_array_equal(out, np.array([1, 2, 3, 4]))

    def test_series(self):
        from nkululeko.predict import _flatten_features

        out = _flatten_features(pd.Series([1, 2, 3]))
        np.testing.assert_array_equal(out, np.array([1, 2, 3]))

    def test_dataframe(self):
        from nkululeko.predict import _flatten_features

        out = _flatten_features(pd.DataFrame({"a": [1], "b": [2]}))
        assert out.shape == (2,)

    def test_list(self):
        from nkululeko.predict import _flatten_features

        np.testing.assert_array_equal(
            _flatten_features([1.0, 2.0]), np.array([1.0, 2.0])
        )


class TestBuildSegmentedDf:
    """`_build_segmented_df` produces a 0-column audformat-segmented frame."""

    def test_segmented_index(self, tmp_path):
        from nkululeko.predict import _build_segmented_df

        a = tmp_path / "a.wav"
        b = tmp_path / "b.wav"
        _write_silent_wav(a)
        _write_silent_wav(b)

        df = _build_segmented_df([str(a), str(b)])
        assert df.shape == (2, 0)
        assert audformat.is_segmented_index(df.index)
        files = df.index.get_level_values("file").tolist()
        assert files == [str(a), str(b)]

    def test_relative_paths_made_absolute(self, tmp_path, monkeypatch):
        from nkululeko.predict import _build_segmented_df

        monkeypatch.chdir(tmp_path)
        _write_silent_wav("a.wav")
        df = _build_segmented_df(["a.wav"])
        path = df.index.get_level_values("file")[0]
        assert os.path.isabs(path)

    def test_nat_ends_resolved_to_real_duration(self, tmp_path):
        """Regression: feature extractors emit indices with real end times.
        `_build_segmented_df` must do the same so `Featureset.filter()` can
        match by index equality (otherwise predictions become NaN)."""
        from nkululeko.predict import _build_segmented_df

        wav = tmp_path / "x.wav"
        _write_silent_wav(wav, samples=16000, sr=16000)  # exactly 1 s

        df = _build_segmented_df([str(wav)])
        end = df.index.get_level_values("end")[0]
        assert not pd.isna(end)
        assert end == pd.Timedelta(seconds=1)


class TestAutopredictTargets:
    """Sanity check for the autopredict-target table."""

    def test_standard_targets_present(self):
        from nkululeko.predict import AUTOPREDICT_TARGETS

        for t in (
            "age",
            "gender",
            "emotion",
            "mos",
            "snr",
            "speaker",
            "arousal",
            "valence",
            "dominance",
            "pesq",
            "sdr",
            "stoi",
            "text",
            "translation",
            "textclassification",
        ):
            assert t in AUTOPREDICT_TARGETS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_defaults(self):
        from nkululeko.predict import DEFAULT_OUTFILE, _build_parser

        args = _build_parser().parse_args([])
        assert args.file is None
        assert args.folder is None
        assert args.list_path is None
        assert args.mic is False
        assert args.outfile == DEFAULT_OUTFILE
        assert args.ptype == "feats"
        assert args.config is None
        assert args.model is None
        # Playback is enabled by default in --mic mode.
        assert args.no_playback is False

    def test_no_playback_flag(self):
        from nkululeko.predict import _build_parser

        args = _build_parser().parse_args(["--no_playback"])
        assert args.no_playback is True

    def test_language_flag_defaults_to_none(self):
        from nkululeko.predict import _build_parser

        args = _build_parser().parse_args([])
        assert args.language is None

    def test_language_flag_accepts_iso_code(self):
        from nkululeko.predict import _build_parser

        args = _build_parser().parse_args(["--language", "de"])
        assert args.language == "de"

    def test_file_accepts_multiple_values(self):
        from nkululeko.predict import _build_parser

        args = _build_parser().parse_args(["--file", "a.wav", "b.wav"])
        assert args.file == ["a.wav", "b.wav"]

    def test_list_path_attribute_name(self):
        """--list maps to the `list_path` attribute (since `list` is reserved)."""
        from nkululeko.predict import _build_parser

        args = _build_parser().parse_args(["--list", "in.csv"])
        assert args.list_path == "in.csv"

    def test_mutually_exclusive_sources(self):
        from nkululeko.predict import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--file", "a.wav", "--mic"])

    def test_type_choice_validated(self):
        from nkululeko.predict import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--type", "bogus"])

    def test_full_argument_set(self):
        from nkululeko.predict import _build_parser

        args = _build_parser().parse_args(
            [
                "--list",
                "in.csv",
                "--config",
                "exp.ini",
                "--type",
                "model",
                "--outfile",
                "out.csv",
                "--model",
                "emotion",
            ]
        )
        assert args.list_path == "in.csv"
        assert args.config == "exp.ini"
        assert args.ptype == "model"
        assert args.outfile == "out.csv"
        assert args.model == "emotion"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_default_config_when_no_arg(self):
        import ast

        from nkululeko.predict import _load_config

        args = argparse.Namespace(config=None)
        config = _load_config(args)
        for section in ("EXP", "DATA", "FEATS", "MODEL"):
            assert section in config
        # DATA.databases must remain parseable for downstream consumers.
        ast.literal_eval(config["DATA"]["databases"])
        # no_reuse is the safe default for ad-hoc runs (no cache reuse).
        assert config["FEATS"]["no_reuse"] == "True"

    def test_loads_explicit_config_and_adds_missing_sections(self, tmp_path):
        from nkululeko.predict import _load_config

        src = configparser.ConfigParser()
        src["EXP"] = {"root": "./", "name": "x"}
        src["DATA"] = {"target": "emotion"}
        p = tmp_path / "x.ini"
        with open(p, "w") as fh:
            src.write(fh)

        args = argparse.Namespace(config=str(p))
        loaded = _load_config(args)
        assert loaded["EXP"]["name"] == "x"
        assert "FEATS" in loaded
        assert "MODEL" in loaded
        # databases gets a default when not provided
        assert "databases" in loaded["DATA"]

    def test_nonexistent_config_exits(self):
        from nkululeko.predict import _load_config

        args = argparse.Namespace(config="/does/not/exist.ini")
        with pytest.raises(SystemExit):
            _load_config(args)

    def test_tmp_root_created_and_atexit_cleans_up(self):
        """When --config is omitted, _load_config creates a temp root and
        registers an atexit handler to delete it. We can't easily wait for
        interpreter shutdown, so we exercise the registered cleanup helper
        directly to confirm the directory is removed."""
        import atexit
        import tempfile as _tempfile

        from nkululeko.predict import _cleanup_path, _load_config

        args = argparse.Namespace(config=None)
        config = _load_config(args)
        tmp_root = config["EXP"]["root"]
        try:
            assert os.path.isdir(tmp_root)
            assert tmp_root.startswith(_tempfile.gettempdir())
            # Confirm the atexit handler will clean it up.
            _cleanup_path(tmp_root)
            assert not os.path.exists(tmp_root)
        finally:
            # Unregister so the suite-shutdown atexit doesn't fire on a
            # stale path from this test.
            atexit.unregister(_cleanup_path)

    def test_cleanup_path_missing_dir_is_noop(self):
        """`_cleanup_path` on a missing path must not raise."""
        from nkululeko.predict import _cleanup_path

        _cleanup_path("/no/such/path/that/should/exist")  # should not raise

    def test_cleanup_path_none_is_noop(self):
        from nkululeko.predict import _cleanup_path

        _cleanup_path(None)  # should not raise

    def test_explicit_config_does_not_create_tmp_root(self, tmp_path):
        """`_load_config` with an explicit --config must not create a temp
        root nor register a cleanup."""
        import atexit

        from nkululeko.predict import _load_config

        cfg_path = tmp_path / "x.ini"
        cfg_path.write_text("[EXP]\nname = x\nroot = ./\n")

        args = argparse.Namespace(config=str(cfg_path))
        # If a cleanup were wrongly registered, the assertion below would
        # not catch it directly — but the EXP.root won't point at a temp
        # dir, which is the observable contract.
        config = _load_config(args)
        assert config["EXP"]["root"] == "./"
        # No temp dir should have been created when an explicit config is supplied.   
        assert "nkulu_predict_" not in config["EXP"]["root"] 


class TestApplyLanguageOverride:
    """`_apply_language_override` writes the CLI value into both config keys."""

    def test_sets_both_keys(self):
        from nkululeko.predict import _apply_language_override

        cfg = configparser.ConfigParser()
        cfg["EXP"] = {"name": "x"}
        _apply_language_override(cfg, "de")
        assert cfg["EXP"]["language"] == "de"
        assert cfg["PREDICT"]["target_language"] == "de"

    def test_creates_missing_sections(self):
        from nkululeko.predict import _apply_language_override

        cfg = configparser.ConfigParser()
        _apply_language_override(cfg, "pl")
        assert "EXP" in cfg
        assert "PREDICT" in cfg
        assert cfg["EXP"]["language"] == "pl"
        assert cfg["PREDICT"]["target_language"] == "pl"

    def test_overrides_existing_values(self):
        from nkululeko.predict import _apply_language_override

        cfg = configparser.ConfigParser()
        cfg["EXP"] = {"language": "de"}
        cfg["PREDICT"] = {"target_language": "de"}
        _apply_language_override(cfg, "en")
        assert cfg["EXP"]["language"] == "en"
        assert cfg["PREDICT"]["target_language"] == "en"


class TestMainLanguageOverride:
    """End-to-end: `--language` writes into glob_conf.config before dispatch."""

    def test_main_propagates_language_into_glob_conf(self, tmp_path):
        from nkululeko import predict as predict_mod
        import nkululeko.glob_conf as glob_conf

        # Real wav so _build_segmented_df can read its duration.
        wav = tmp_path / "x.wav"
        _write_silent_wav(wav)

        # Capture glob_conf.config at the moment _predict_df is called.
        captured = {}

        def fake_predict_df(seg_df, args, util):
            captured["language"] = glob_conf.config["EXP"]["language"]
            captured["target_language"] = glob_conf.config["PREDICT"]["target_language"]
            return pd.DataFrame({"text": ["hi"]}, index=seg_df.index)

        argv = [
            "predict.py",
            "--file",
            str(wav),
            "--model",
            "text",
            "--language",
            "de",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(predict_mod, "_predict_df", side_effect=fake_predict_df),
        ):
            predict_mod.main()

        assert captured["language"] == "de"
        assert captured["target_language"] == "de"


# ---------------------------------------------------------------------------
# _run_files
# ---------------------------------------------------------------------------


class TestRunFiles:
    """`_run_files` writes a `<name>_result.txt` per valid input file."""

    def test_per_file_output(self, tmp_path):
        f1 = tmp_path / "one.wav"
        f2 = tmp_path / "two.wav"
        _write_silent_wav(f1)
        _write_silent_wav(f2)

        def fake_predict_df(seg_df, args, util):
            return pd.DataFrame(
                {"emotion_pred": ["happy", "sad"]},
                index=seg_df.index,
            )

        from nkululeko.predict import _run_files

        util = MagicMock()
        with patch("nkululeko.predict._predict_df", side_effect=fake_predict_df):
            _run_files([str(f1), str(f2)], argparse.Namespace(), util)

        r1 = tmp_path / "one_result.txt"
        r2 = tmp_path / "two_result.txt"
        assert r1.is_file()
        assert r2.is_file()
        assert "emotion_pred: happy" in r1.read_text()
        assert "emotion_pred: sad" in r2.read_text()

    def test_multi_column_predictions(self, tmp_path):
        """Every prediction column is written to the per-file output."""
        audio = tmp_path / "x.wav"
        _write_silent_wav(audio)

        def fake_predict_df(seg_df, args, util):
            return pd.DataFrame(
                {"angry": [0.8], "happy": [0.1], "predicted": ["angry"]},
                index=seg_df.index,
            )

        from nkululeko.predict import _run_files

        util = MagicMock()
        with patch("nkululeko.predict._predict_df", side_effect=fake_predict_df):
            _run_files([str(audio)], argparse.Namespace(), util)

        body = (tmp_path / "x_result.txt").read_text()
        assert "angry: 0.8" in body
        assert "happy: 0.1" in body
        assert "predicted: angry" in body

    def test_skips_missing_files(self, tmp_path):
        f1 = tmp_path / "real.wav"
        _write_silent_wav(f1)
        missing = tmp_path / "ghost.wav"

        def fake_predict_df(seg_df, args, util):
            return pd.DataFrame({"col": ["x"]}, index=seg_df.index)

        from nkululeko.predict import _run_files

        util = MagicMock()
        with patch("nkululeko.predict._predict_df", side_effect=fake_predict_df):
            _run_files([str(f1), str(missing)], argparse.Namespace(), util)

        assert util.warn.called
        assert (tmp_path / "real_result.txt").is_file()
        assert not (tmp_path / "ghost_result.txt").is_file()

    def test_no_valid_files_calls_error(self):
        from nkululeko.predict import _run_files

        util = MagicMock()
        util.error.side_effect = SystemExit
        with pytest.raises(SystemExit):
            _run_files(["/nope/missing.wav"], argparse.Namespace(), util)


# ---------------------------------------------------------------------------
# _run_list
# ---------------------------------------------------------------------------


class TestRunList:
    def test_preserves_original_columns(self, tmp_path):
        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        _write_silent_wav(f1)
        _write_silent_wav(f2)

        csv = tmp_path / "in.csv"
        pd.DataFrame(
            {
                "file": [str(f1), str(f2)],
                "speaker": ["s1", "s2"],
                "note": ["x", "y"],
            }
        ).to_csv(csv, index=False)

        outfile = tmp_path / "out.csv"

        def fake_predict_df(seg_df, args, util):
            return pd.DataFrame(
                {"snr_pred": [1.5, 2.5]},
                index=seg_df.index,
            )

        from nkululeko.predict import _run_list

        args = argparse.Namespace(outfile=str(outfile))
        util = MagicMock()
        with patch("nkululeko.predict._predict_df", side_effect=fake_predict_df):
            _run_list(str(csv), args, util)

        result = pd.read_csv(outfile)
        # Original columns survive, prediction column is appended.
        assert "speaker" in result.columns
        assert "note" in result.columns
        assert "snr_pred" in result.columns
        assert sorted(result["speaker"].tolist()) == ["s1", "s2"]

    def test_missing_csv_errors(self):
        from nkululeko.predict import _run_list

        util = MagicMock()
        util.error.side_effect = SystemExit
        with pytest.raises(SystemExit):
            _run_list("/no/such.csv", argparse.Namespace(outfile="x.csv"), util)

    def test_single_column_file_list(self, tmp_path):
        """A CSV with only a `file` column is parsed by audformat as an Index
        (not a DataFrame); _run_list must handle that case."""
        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        _write_silent_wav(f1)
        _write_silent_wav(f2)

        csv = tmp_path / "files_only.csv"
        with open(csv, "w") as fh:
            fh.write("file\n")
            fh.write(f"{f1}\n")
            fh.write(f"{f2}\n")

        outfile = tmp_path / "out.csv"

        def fake_predict_df(seg_df, args, util):
            return pd.DataFrame(
                {"snr_pred": [1.0, 2.0]},
                index=seg_df.index,
            )

        from nkululeko.predict import _run_list

        args = argparse.Namespace(outfile=str(outfile))
        util = MagicMock()
        with patch("nkululeko.predict._predict_df", side_effect=fake_predict_df):
            _run_list(str(csv), args, util)

        result = pd.read_csv(outfile)
        assert "snr_pred" in result.columns
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _run_folder
# ---------------------------------------------------------------------------


class TestRunFolder:
    def test_empty_folder_calls_error(self, tmp_path):
        from nkululeko.predict import _run_folder

        util = MagicMock()
        util.error.side_effect = SystemExit
        with pytest.raises(SystemExit):
            _run_folder(str(tmp_path), argparse.Namespace(outfile="x.csv"), util)

    def test_nonexistent_folder_calls_error(self):
        from nkululeko.predict import _run_folder

        util = MagicMock()
        util.error.side_effect = SystemExit
        with pytest.raises(SystemExit):
            _run_folder("/no/such/folder", argparse.Namespace(outfile="x.csv"), util)

    def test_finds_audio_and_writes_csv(self, tmp_path):
        _write_silent_wav(tmp_path / "a.wav")
        _write_silent_wav(tmp_path / "b.wav")
        (tmp_path / "notes.txt").touch()  # ignored extension

        out = tmp_path / "out.csv"

        def fake_predict_df(seg_df, args, util):
            return pd.DataFrame(
                {"x_pred": list(range(len(seg_df)))},
                index=seg_df.index,
            )

        from nkululeko.predict import _run_folder

        util = MagicMock()
        args = argparse.Namespace(outfile=str(out))
        with patch("nkululeko.predict._predict_df", side_effect=fake_predict_df):
            _run_folder(str(tmp_path), args, util)

        df = pd.read_csv(out)
        # 2 audio files (txt excluded)
        assert len(df) == 2
        assert "x_pred" in df.columns


# ---------------------------------------------------------------------------
# _run_mic
# ---------------------------------------------------------------------------


def _run_mic_with_mocks(args, monkeypatch):
    """Invoke `_run_mic` with one recording iteration and quit.

    Returns the mocked `sounddevice` module so callers can inspect calls.
    """
    sd_mock = MagicMock()
    sd_mock.rec.return_value = np.zeros((1600, 1), dtype="float32")
    sf_mock = MagicMock()
    monkeypatch.setitem(sys.modules, "sounddevice", sd_mock)
    monkeypatch.setitem(sys.modules, "soundfile", sf_mock)

    # First input() returns "" (record once), second returns "q" (quit).
    answers = iter(["", "q"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))

    # Skip touching real files / extractors.
    util = MagicMock()
    from nkululeko import predict as predict_mod

    monkeypatch.setattr(
        predict_mod,
        "_build_segmented_df",
        lambda files: pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                [(files[0], pd.Timedelta(0), pd.Timedelta(seconds=1))],
                names=["file", "start", "end"],
            )
        ),
    )
    monkeypatch.setattr(
        predict_mod,
        "_predict_df",
        lambda seg_df, args, util: pd.DataFrame({"age_pred": [42]}, index=seg_df.index),
    )
    monkeypatch.setattr(os, "remove", lambda *a, **kw: None)

    predict_mod._run_mic(args, util)
    return sd_mock


class TestRunMic:
    def test_playback_by_default(self, monkeypatch):
        args = argparse.Namespace(no_playback=False)
        sd = _run_mic_with_mocks(args, monkeypatch)
        assert sd.rec.called
        assert sd.play.called  # playback ran

    def test_no_playback_when_flag_set(self, monkeypatch):
        args = argparse.Namespace(no_playback=True)
        sd = _run_mic_with_mocks(args, monkeypatch)
        assert sd.rec.called
        assert not sd.play.called  # playback suppressed

    def test_playback_failure_warns_but_continues(self, monkeypatch):
        """A playback exception must be caught — prediction still runs."""
        sd_mock = MagicMock()
        sd_mock.rec.return_value = np.zeros((1600, 1), dtype="float32")
        sd_mock.play.side_effect = RuntimeError("no audio device")
        sf_mock = MagicMock()
        monkeypatch.setitem(sys.modules, "sounddevice", sd_mock)
        monkeypatch.setitem(sys.modules, "soundfile", sf_mock)

        answers = iter(["", "q"])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))

        from nkululeko import predict as predict_mod

        monkeypatch.setattr(
            predict_mod,
            "_build_segmented_df",
            lambda files: pd.DataFrame(
                index=pd.MultiIndex.from_tuples(
                    [(files[0], pd.Timedelta(0), pd.Timedelta(seconds=1))],
                    names=["file", "start", "end"],
                )
            ),
        )
        predict_mock = MagicMock(
            return_value=pd.DataFrame(
                {"age_pred": [42]},
                index=pd.MultiIndex.from_tuples(
                    [("x", pd.Timedelta(0), pd.Timedelta(seconds=1))],
                    names=["file", "start", "end"],
                ),
            )
        )
        monkeypatch.setattr(predict_mod, "_predict_df", predict_mock)
        monkeypatch.setattr(os, "remove", lambda *a, **kw: None)

        util = MagicMock()
        args = argparse.Namespace(no_playback=False)
        predict_mod._run_mic(args, util)

        # Prediction ran despite playback failure; util.warn was invoked.
        assert predict_mock.called
        assert util.warn.called


# ---------------------------------------------------------------------------
# _predict_df dispatch
# ---------------------------------------------------------------------------


class TestPredictDfDispatch:
    def test_model_path(self):
        from nkululeko.predict import _predict_df

        args = argparse.Namespace(ptype="model", model=None)
        util = MagicMock()
        with patch(
            "nkululeko.predict._predict_with_model", return_value="MODEL"
        ) as mock:
            assert _predict_df(pd.DataFrame(), args, util) == "MODEL"
            mock.assert_called_once()

    def test_autopredict_path(self):
        from nkululeko.predict import _predict_df

        args = argparse.Namespace(ptype="feats", model="emotion")
        util = MagicMock()
        with patch(
            "nkululeko.predict._predict_with_autopredict", return_value="AP"
        ) as mock:
            assert _predict_df(pd.DataFrame(), args, util) == "AP"
            mock.assert_called_once()
            # target arg is the lowercased model name
            assert mock.call_args.args[1] == "emotion"

    def test_autopredict_case_insensitive(self):
        from nkululeko.predict import _predict_df

        args = argparse.Namespace(ptype="feats", model="EMOTION")
        util = MagicMock()
        with patch(
            "nkululeko.predict._predict_with_autopredict", return_value="AP"
        ) as mock:
            _predict_df(pd.DataFrame(), args, util)
            assert mock.call_args.args[1] == "emotion"

    def test_feature_extractor_fallback(self):
        from nkululeko.predict import _predict_df

        args = argparse.Namespace(ptype="feats", model="wav2vec2")
        util = MagicMock()
        with patch(
            "nkululeko.predict._predict_with_features", return_value="FEAT"
        ) as mock:
            assert _predict_df(pd.DataFrame(), args, util) == "FEAT"
            mock.assert_called_once()

    def test_overlapping_name_resolves_to_autopredict(self):
        """`mos` and `snr` are both autopredict targets and extractor names —
        the dispatch must prefer the autopredict path."""
        from nkululeko.predict import _predict_df

        util = MagicMock()
        for name in ("mos", "snr"):
            args = argparse.Namespace(ptype="feats", model=name)
            with (
                patch(
                    "nkululeko.predict._predict_with_autopredict", return_value="AP"
                ) as ap,
                patch("nkululeko.predict._predict_with_features") as feat,
            ):
                _predict_df(pd.DataFrame(), args, util)
                ap.assert_called_once()
                feat.assert_not_called()


# ---------------------------------------------------------------------------
# _get_feature_extractor dispatch
# ---------------------------------------------------------------------------


class TestGetFeatureExtractor:
    """Regression: specific aud* extractors must not be swallowed by the
    generic `wav2vec2 in name` / `wav2vec in name` substring checks."""

    def test_audwav2vec2_dispatches_to_audwav2vec2set(self, monkeypatch):
        """Bug: `audwav2vec2` was matched by `"wav2vec2" in name` and routed
        to the generic `Wav2vec2` loader, which then tried to fetch the
        bogus HF model `facebook/audwav2vec2` and crashed."""
        from nkululeko.predict import _get_feature_extractor

        # Track which extractor class was instantiated.
        captured = {}

        class FakeAudwav2vec2Set:
            def __init__(self, name, data, ftype):
                captured["cls"] = "audwav2vec2"

            def _load_model(self):
                pass

        class FakeWav2vec2:
            def __init__(self, *a, **kw):
                captured["cls"] = "wav2vec2"

            def init_model(self):
                pass

        # Replace both modules so we can tell which branch was taken without
        # actually downloading any models.
        from nkululeko.feat_extract import feats_audwav2vec2, feats_wav2vec2

        monkeypatch.setattr(feats_audwav2vec2, "Audwav2vec2Set", FakeAudwav2vec2Set)
        monkeypatch.setattr(feats_wav2vec2, "Wav2vec2", FakeWav2vec2)

        _get_feature_extractor("audwav2vec2", MagicMock())
        assert captured["cls"] == "audwav2vec2"

    def test_auddim_dispatches_to_auddimset(self, monkeypatch):
        from nkululeko.predict import _get_feature_extractor

        captured = {}

        class FakeAuddimSet:
            def __init__(self, *a, **kw):
                captured["cls"] = "auddim"

            def _load_model(self):
                pass

        from nkululeko.feat_extract import feats_auddim

        monkeypatch.setattr(feats_auddim, "AuddimSet", FakeAuddimSet)
        _get_feature_extractor("auddim", MagicMock())
        assert captured["cls"] == "auddim"

    def test_plain_wav2vec2_still_dispatches_to_generic(self, monkeypatch):
        """Ensure the fix didn't accidentally divert plain `wav2vec2` names."""
        from nkululeko.predict import _get_feature_extractor

        captured = {}

        class FakeWav2vec2:
            def __init__(self, *a, **kw):
                captured["cls"] = "wav2vec2"

            def init_model(self):
                pass

        from nkululeko.feat_extract import feats_wav2vec2

        monkeypatch.setattr(feats_wav2vec2, "Wav2vec2", FakeWav2vec2)
        _get_feature_extractor("wav2vec2-large-robust", MagicMock())
        assert captured["cls"] == "wav2vec2"

    def test_unknown_extractor_calls_error(self):
        from nkululeko.predict import _get_feature_extractor

        util = MagicMock()
        util.error.side_effect = SystemExit
        with pytest.raises(SystemExit):
            _get_feature_extractor("totally-unknown-thing", util)


# ---------------------------------------------------------------------------
# _predict_with_model
# ---------------------------------------------------------------------------


class TestPredictWithModel:
    """Regression: --type model must build a fresh feature extractor from
    FEATS.type, not use the (lie-prone) pickled extractor on the experiment.

    Background: experiment.save() strips inner model/model_interface fields
    so the object can be pickled, but leaves `model_loaded=True` behind.
    A pickled extractor therefore reports as "loaded" but would AttributeError
    in extract_sample(). We must always re-instantiate."""

    def test_missing_feats_type_calls_error(self, monkeypatch, tmp_path):
        from nkululeko.predict import _predict_with_model

        # Minimal in-memory config without FEATS.type set.
        import configparser

        import nkululeko.glob_conf as glob_conf

        cfg = configparser.ConfigParser()
        cfg["EXP"] = {"root": "./", "name": "x"}
        cfg["DATA"] = {"databases": "['adhoc']"}
        cfg["FEATS"] = {}
        cfg["MODEL"] = {}
        monkeypatch.setattr(glob_conf, "config", cfg)

        # Stub out Experiment to avoid pickle / filesystem access.
        from nkululeko import predict as predict_mod

        fake_expr = MagicMock()
        monkeypatch.setattr(
            predict_mod,
            "Experiment",
            lambda *a, **kw: fake_expr,
            raising=False,
        )
        # Patch the import inside _predict_with_model.
        import nkululeko.experiment as expmod

        monkeypatch.setattr(expmod, "Experiment", lambda *a, **kw: fake_expr)

        util = MagicMock()
        util.error.side_effect = SystemExit
        util.get_save_name.return_value = str(tmp_path / "does_not_matter")

        with pytest.raises(SystemExit):
            _predict_with_model(pd.DataFrame(), argparse.Namespace(), util)

    def test_builds_fresh_extractor_from_feats_type(self, monkeypatch, tmp_path):
        """`_predict_with_model` should call `_get_feature_extractor(name)`
        where `name` is the first entry of `FEATS.type`."""
        import configparser

        import nkululeko.glob_conf as glob_conf
        from nkululeko import predict as predict_mod

        # Real audio file so duration-resolution works upstream.
        wav = tmp_path / "x.wav"
        _write_silent_wav(wav)

        cfg = configparser.ConfigParser()
        cfg["EXP"] = {"root": str(tmp_path), "name": "x"}
        cfg["DATA"] = {"databases": "['adhoc']", "target": "emotion"}
        cfg["FEATS"] = {"type": "['praat']"}
        cfg["MODEL"] = {}
        monkeypatch.setattr(glob_conf, "config", cfg)

        # Fake experiment with the bits _predict_with_model uses.
        fake_model = MagicMock()
        fake_model.predict_sample.return_value = {0: 0.7, 1: 0.3}
        fake_expr = MagicMock()
        fake_expr.runmgr.get_best_model.return_value = fake_model
        # Sentinel value on the pickled feature_extractor — we should NOT use it.
        fake_expr.feature_extractor = MagicMock(name="PICKLED-EXTRACTOR-UNUSED")
        fake_expr.label_encoder = None
        import nkululeko.experiment as expmod

        monkeypatch.setattr(expmod, "Experiment", lambda *a, **kw: fake_expr)

        # Spy on _get_feature_extractor.
        fresh_extractor = MagicMock()
        fresh_extractor.extract_sample.return_value = np.array([1.0, 2.0])
        spy = MagicMock(return_value=fresh_extractor)
        monkeypatch.setattr(predict_mod, "_get_feature_extractor", spy)

        # Build the input.
        seg_df = predict_mod._build_segmented_df([str(wav)])

        util = MagicMock()
        util.get_save_name.return_value = str(tmp_path / "whatever")
        util.exp_is_classification.return_value = True
        util.config_val.side_effect = lambda section, key, default: (
            cfg[section][key] if section in cfg and key in cfg[section] else default
        )

        preds = predict_mod._predict_with_model(seg_df, argparse.Namespace(), util)

        # Spy was called with the FEATS.type entry, not "PICKLED-EXTRACTOR-UNUSED".
        assert spy.called
        assert spy.call_args.args[0] == "praat"
        # Pickled extractor was not consulted.
        fake_expr.feature_extractor.extract_sample.assert_not_called()
        # The fresh extractor produced features and we got back a prediction row.
        assert len(preds) == 1


# ---------------------------------------------------------------------------
# do_test backwards-compat shim
# ---------------------------------------------------------------------------


class TestDoTest:
    def test_missing_config_exits(self):
        from nkululeko.predict import do_test

        with pytest.raises(SystemExit):
            do_test("/no/such/config.ini", "out.csv")


# ---------------------------------------------------------------------------
# _run_from_config
# ---------------------------------------------------------------------------


def _make_segmented_df(files):
    """Build a small audformat-segmented DataFrame for fake df_train/df_test."""
    starts = [pd.Timedelta(0)] * len(files)
    ends = [pd.Timedelta(seconds=1)] * len(files)
    idx = audformat.segmented_index(files=files, starts=starts, ends=ends)
    return pd.DataFrame({"emotion": ["happy"] * len(files)}, index=idx)


class _FakeExperimentFactory:
    """Builds a fake Experiment with the methods _run_from_config calls."""

    def __init__(self, df_train, df_test):
        self._df_train = df_train
        self._df_test = df_test

    def __call__(self, config):
        e = MagicMock()
        e.df_train = self._df_train
        e.df_test = self._df_test
        return e


class TestRunFromConfig:
    """`_run_from_config` reads EXP.sample_selection (defaults to 'all'),
    runs the prediction over the selected subset, and writes one CSV."""

    def _run(self, monkeypatch, tmp_path, sample_selection=None, n_train=2, n_test=3):
        """Drive _run_from_config with a stubbed Experiment + predict_df."""
        import configparser

        import nkululeko.glob_conf as glob_conf
        from nkululeko import predict as predict_mod
        import nkululeko.experiment as expmod

        # Real wavs so segmented indices are valid; n_train+n_test files.
        train_files, test_files = [], []
        for i in range(n_train):
            p = tmp_path / f"train_{i}.wav"
            _write_silent_wav(p)
            train_files.append(str(p))
        for i in range(n_test):
            p = tmp_path / f"test_{i}.wav"
            _write_silent_wav(p)
            test_files.append(str(p))

        df_train = _make_segmented_df(train_files)
        df_test = _make_segmented_df(test_files)
        monkeypatch.setattr(
            expmod, "Experiment", _FakeExperimentFactory(df_train, df_test)
        )

        # Minimal config; EXP.sample_selection only set when caller asks.
        cfg = configparser.ConfigParser()
        cfg["EXP"] = {"name": "x", "root": str(tmp_path)}
        if sample_selection is not None:
            cfg["EXP"]["sample_selection"] = sample_selection
        cfg["DATA"] = {"databases": "['adhoc']"}
        cfg["FEATS"] = {}
        cfg["MODEL"] = {}
        monkeypatch.setattr(glob_conf, "config", cfg)

        # Capture the seg_df passed to _predict_df so we can assert on size.
        seen = {}

        def fake_predict_df(seg_df, args, util):
            seen["n_rows"] = len(seg_df)
            return pd.DataFrame({"snr_pred": [1.0] * len(seg_df)}, index=seg_df.index)

        monkeypatch.setattr(predict_mod, "_predict_df", fake_predict_df)

        out = tmp_path / "out.csv"
        args = argparse.Namespace(
            config="dummy.ini",
            outfile=str(out),
            ptype="feats",
            model="snr",
        )
        util = MagicMock()
        util.error.side_effect = SystemExit
        util.config_val.side_effect = lambda section, key, default: (
            cfg[section][key] if section in cfg and key in cfg[section] else default
        )

        predict_mod._run_from_config(args, util)
        return seen, out

    def test_defaults_to_all(self, monkeypatch, tmp_path):
        seen, out = self._run(
            monkeypatch, tmp_path, sample_selection=None, n_train=2, n_test=3
        )
        # default = all -> concatenated rows
        assert seen["n_rows"] == 5
        assert out.is_file()
        # Output has the prediction column merged in.
        result = pd.read_csv(out)
        assert "snr_pred" in result.columns

    def test_sample_selection_all(self, monkeypatch, tmp_path):
        seen, _ = self._run(
            monkeypatch, tmp_path, sample_selection="all", n_train=2, n_test=3
        )
        assert seen["n_rows"] == 5

    def test_sample_selection_train(self, monkeypatch, tmp_path):
        seen, _ = self._run(
            monkeypatch, tmp_path, sample_selection="train", n_train=2, n_test=3
        )
        assert seen["n_rows"] == 2

    def test_sample_selection_test(self, monkeypatch, tmp_path):
        seen, _ = self._run(
            monkeypatch, tmp_path, sample_selection="test", n_train=2, n_test=3
        )
        assert seen["n_rows"] == 3

    def test_unknown_sample_selection_errors(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit):
            self._run(monkeypatch, tmp_path, sample_selection="bogus")

    def test_empty_selection_errors(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit):
            self._run(
                monkeypatch, tmp_path, sample_selection="train", n_train=0, n_test=3
            )


class TestMainFallthroughToConfig:
    """End-to-end: --config without any input flag dispatches to _run_from_config."""

    def test_main_with_only_config_calls_run_from_config(self, tmp_path, monkeypatch):
        from nkululeko import predict as predict_mod

        # Write a real config file so _load_config can read it.
        cfg_path = tmp_path / "exp.ini"
        cfg_path.write_text(
            "[EXP]\nroot = ./\nname = x\nsample_selection = train\n"
            "[DATA]\ndatabases = ['adhoc']\n"
            "[FEATS]\ntype = ['snr']\n"
            "[MODEL]\n"
        )

        called = {"yes": False}

        def fake_run_from_config(args, util):
            called["yes"] = True
            # match the EXP.sample_selection we set above
            assert util.config_val("EXP", "sample_selection", "all") == "train"

        monkeypatch.setattr(predict_mod, "_run_from_config", fake_run_from_config)

        argv = ["predict.py", "--config", str(cfg_path)]
        with patch.object(sys, "argv", argv):
            predict_mod.main()

        assert called["yes"]


# ---------------------------------------------------------------------------
# main() smoke
# ---------------------------------------------------------------------------


class TestMain:
    def test_help_exits_cleanly(self):
        from nkululeko.predict import main

        with patch.object(sys, "argv", ["predict.py", "--help"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0

    def test_no_input_and_no_config_errors(self):
        """Without --file/--folder/--list/--mic AND without --config,
        util.error is invoked. (With --config we fall through to
        _run_from_config; see TestRunFromConfig.)"""
        from nkululeko.predict import main

        # `--model snr` satisfies the FEATS-type pre-check so we hit the
        # "no input given" branch (util.error -> sys.exit).
        with patch.object(sys, "argv", ["predict.py", "--model", "snr"]):
            with pytest.raises(SystemExit):
                main()

    def test_type_model_requires_config(self):
        """`--type model` without `--config` must call util.error -> exit."""
        from nkululeko.predict import main

        with patch.object(
            sys,
            "argv",
            ["predict.py", "--type", "model", "--file", "x.wav"],
        ):
            with pytest.raises(SystemExit):
                main()


# ---------------------------------------------------------------------------
# Integration through main() with backends mocked
# ---------------------------------------------------------------------------


class TestMainEndToEnd:
    def test_main_dispatches_to_files(self, tmp_path):
        """End-to-end: main() drives _run_files for --file, given a stub backend."""
        from nkululeko import predict as predict_mod

        audio = tmp_path / "sample.wav"
        _write_silent_wav(audio)

        def fake_predict_df(seg_df, args, util):
            return pd.DataFrame(
                {"snr_pred": [42.0]},
                index=seg_df.index,
            )

        argv = [
            "predict.py",
            "--file",
            str(audio),
            "--model",
            "snr",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(predict_mod, "_predict_df", side_effect=fake_predict_df),
        ):
            predict_mod.main()

        out = tmp_path / "sample_result.txt"
        assert out.is_file()
        assert "snr_pred: 42.0" in out.read_text()
