"""Tests for the nkululeko.avqi module.

Covers:

- `compute_avqi` (parselmouth mocked)
- `_record_clip` (sounddevice mocked): accept/re-record, playback toggling
- `run_interactive`: reuse of existing --sv/--cs files vs. recording
- `main()` argument parsing and validation
"""

import argparse
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nkululeko import avqi as avqi_mod
from nkululeko.utils.errors import NkululukoError


# ---------------------------------------------------------------------------
# compute_avqi
# ---------------------------------------------------------------------------


class TestComputeAvqi:
    def test_returns_expected_keys(self, tmp_path, monkeypatch):
        variables = {
            "cpps": 10.0,
            "hnr": 20.0,
            "shim": 1.5,
            "shdb": 0.3,
            "slope": -10.0,
            "tilt": -5.0,
            "avqi": 3.14,
        }
        parselmouth_mock = MagicMock()
        parselmouth_mock.praat.run.return_value = (MagicMock(), variables)
        monkeypatch.setitem(sys.modules, "parselmouth", parselmouth_mock)

        sv = tmp_path / "sv.wav"
        cs = tmp_path / "cs.wav"
        sv.touch()
        cs.touch()

        result = avqi_mod.compute_avqi(str(sv), str(cs))

        assert result == {
            "cpps": 10.0,
            "hnr": 20.0,
            "shimmer_local": 1.5,
            "shimmer_local_db": 0.3,
            "ltas_slope": -10.0,
            "ltas_tilt": -5.0,
            "avqi": 3.14,
        }
        # paths passed to praat.run should be absolute
        call_args = parselmouth_mock.praat.run.call_args
        assert os.path.isabs(call_args[0][1])
        assert os.path.isabs(call_args[0][2])

    def test_handles_plain_dict_return(self, tmp_path, monkeypatch):
        """Some parselmouth versions return the variables dict directly."""
        variables = {
            "cpps": 1.0,
            "hnr": 2.0,
            "shim": 3.0,
            "shdb": 4.0,
            "slope": 5.0,
            "tilt": 6.0,
            "avqi": 7.0,
        }
        parselmouth_mock = MagicMock()
        parselmouth_mock.praat.run.return_value = variables
        monkeypatch.setitem(sys.modules, "parselmouth", parselmouth_mock)

        sv = tmp_path / "sv.wav"
        cs = tmp_path / "cs.wav"
        sv.touch()
        cs.touch()

        result = avqi_mod.compute_avqi(str(sv), str(cs))
        assert result["avqi"] == 7.0


# ---------------------------------------------------------------------------
# _record_clip
# ---------------------------------------------------------------------------


class TestRecordClip:
    def _mock_sd(self, monkeypatch, samples=1600):
        sd_mock = MagicMock()
        sd_mock.rec.return_value = np.zeros((samples, 1), dtype="float32")
        monkeypatch.setitem(sys.modules, "sounddevice", sd_mock)
        return sd_mock

    def test_accepts_recording_on_first_try(self, monkeypatch):
        sd_mock = self._mock_sd(monkeypatch)
        answers = iter(["", "y"])  # Enter to record, "y" to keep
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))
        util = MagicMock()

        signal = avqi_mod._record_clip(1.0, 1600, "prompt", util)

        assert signal.shape == (1600,)
        assert sd_mock.rec.call_count == 1
        assert sd_mock.play.called

    def test_no_playback_skips_playback(self, monkeypatch):
        sd_mock = self._mock_sd(monkeypatch)
        answers = iter(["", ""])  # Enter to record, empty (default yes) to keep
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))
        util = MagicMock()

        avqi_mod._record_clip(1.0, 1600, "prompt", util, no_playback=True)

        assert not sd_mock.play.called

    def test_rerecords_until_accepted(self, monkeypatch):
        sd_mock = self._mock_sd(monkeypatch)
        # record, reject, record, accept
        answers = iter(["", "n", "", "y"])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))
        util = MagicMock()

        avqi_mod._record_clip(1.0, 1600, "prompt", util)

        assert sd_mock.rec.call_count == 2

    def test_playback_failure_warns_but_continues(self, monkeypatch):
        sd_mock = self._mock_sd(monkeypatch)
        sd_mock.play.side_effect = RuntimeError("no audio device")
        answers = iter(["", "y"])
        monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))
        util = MagicMock()

        signal = avqi_mod._record_clip(1.0, 1600, "prompt", util)

        assert signal.shape == (1600,)
        assert util.warn.called


# ---------------------------------------------------------------------------
# run_interactive
# ---------------------------------------------------------------------------


class TestRunInteractive:
    def test_uses_existing_files_without_recording(self, tmp_path, monkeypatch):
        sv = tmp_path / "sv.wav"
        cs = tmp_path / "cs.wav"
        sv.touch()
        cs.touch()

        record_mock = MagicMock()
        monkeypatch.setattr(avqi_mod, "_record_clip", record_mock)
        compute_mock = MagicMock(return_value={"avqi": 1.23})
        monkeypatch.setattr(avqi_mod, "compute_avqi", compute_mock)

        args = argparse.Namespace(
            sv=str(sv),
            cs=str(cs),
            sv_duration=avqi_mod.DEFAULT_SV_DURATION_S,
            cs_duration=avqi_mod.DEFAULT_CS_DURATION_S,
            outdir=None,
            no_playback=False,
        )
        util = MagicMock()

        result = avqi_mod.run_interactive(args, util)

        assert result == {"avqi": 1.23}
        record_mock.assert_not_called()
        compute_mock.assert_called_once_with(str(sv), str(cs))

    def test_records_missing_clips_and_writes_wav(self, tmp_path, monkeypatch):
        record_mock = MagicMock(return_value=np.zeros(1600, dtype="float32"))
        monkeypatch.setattr(avqi_mod, "_record_clip", record_mock)
        compute_mock = MagicMock(return_value={"avqi": 4.56})
        monkeypatch.setattr(avqi_mod, "compute_avqi", compute_mock)

        args = argparse.Namespace(
            sv=None,
            cs=None,
            sv_duration=avqi_mod.DEFAULT_SV_DURATION_S,
            cs_duration=avqi_mod.DEFAULT_CS_DURATION_S,
            outdir=str(tmp_path),
            no_playback=False,
        )
        util = MagicMock()

        result = avqi_mod.run_interactive(args, util)

        assert result == {"avqi": 4.56}
        assert record_mock.call_count == 2
        sv_path = str(tmp_path / "sv.wav")
        cs_path = str(tmp_path / "cs.wav")
        assert os.path.isfile(sv_path)
        assert os.path.isfile(cs_path)
        compute_mock.assert_called_once_with(sv_path, cs_path)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_rejects_short_sv_duration(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["avqi.py", "--sv_duration", "2.0", "--cs", "x.wav"]
        )
        with pytest.raises(NkululukoError):
            avqi_mod.main()

    def test_full_run_with_existing_files(self, tmp_path, monkeypatch):
        sv = tmp_path / "sv.wav"
        cs = tmp_path / "cs.wav"
        sv.touch()
        cs.touch()
        outfile = tmp_path / "result.csv"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "avqi.py",
                "--sv",
                str(sv),
                "--cs",
                str(cs),
                "--outfile",
                str(outfile),
            ],
        )
        compute_mock = MagicMock(return_value={"avqi": 9.9, "hnr": 1, "cpps": 1,
                                                "shimmer_local": 1, "shimmer_local_db": 1,
                                                "ltas_slope": 1, "ltas_tilt": 1})
        monkeypatch.setattr(avqi_mod, "compute_avqi", compute_mock)

        result = avqi_mod.main()

        assert result["avqi"] == 9.9
        assert outfile.is_file()
