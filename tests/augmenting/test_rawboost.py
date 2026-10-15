"""RawBoost (nkululeko/augmenting/rawboost.py) and its offline augmenter
(nkululeko/augmenting/augmenter_rawboost.py)."""

import configparser

import audiofile
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import nkululeko.glob_conf as glob_conf
from nkululeko.augmenting import rawboost as rb
from nkululeko.augmenting.augmenter_rawboost import AugmenterRawBoost, rawboost_signal
from nkululeko.utils.errors import NkululukoError
from nkululeko.utils.util import Util

ALGOS = sorted(rb.ALGORITHMS)


def _signal(sr, seconds=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.5, 0.5, size=int(sr * seconds))


class TestApplyRawboost:
    @pytest.mark.parametrize("sr", [8000, 16000, 44100])
    @pytest.mark.parametrize("algo", ALGOS)
    def test_same_length_and_finite_at_any_sampling_rate(self, algo, sr):
        # the default notch range goes up to 8 kHz, above the Nyquist
        # frequency of 8 kHz audio: repeat with different random notches
        x = _signal(sr)
        for seed in range(8):
            np.random.seed(seed)
            out = rb.apply_rawboost(x, sr, rb.RawBoostParams(algo=algo))
            assert out.shape == x.shape
            assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("algo", ALGOS)
    def test_silence_stays_finite(self, algo):
        out = rb.apply_rawboost(np.zeros(8000), 16000, rb.RawBoostParams(algo=algo))
        assert np.all(np.isfinite(out))

    @pytest.mark.parametrize("algo", ALGOS)
    def test_changes_the_signal(self, algo):
        x = _signal(16000)
        np.random.seed(0)
        out = rb.apply_rawboost(x, 16000, rb.RawBoostParams(algo=algo))
        assert not np.allclose(out, x)

    def test_input_is_not_modified(self):
        x = _signal(16000)
        before = x.copy()
        rb.apply_rawboost(x, 16000, rb.RawBoostParams(algo=4))
        assert np.array_equal(x, before)

    def test_follows_numpys_random_state(self):
        x = _signal(16000)
        params = rb.RawBoostParams(algo=4)
        np.random.seed(3)
        first = rb.apply_rawboost(x, 16000, params)
        np.random.seed(3)
        again = rb.apply_rawboost(x, 16000, params)
        np.random.seed(4)
        other = rb.apply_rawboost(x, 16000, params)
        assert np.array_equal(first, again)
        assert not np.allclose(first, other)

    def test_needs_a_mono_waveform(self):
        with pytest.raises(ValueError, match="mono"):
            rb.apply_rawboost(np.zeros((2, 100)), 16000, rb.RawBoostParams())

    def test_additive_noise_has_the_requested_snr(self):
        x = _signal(16000, seconds=2)
        params = rb.RawBoostParams(algo=3, snr_min=20, snr_max=20)
        np.random.seed(0)
        noise = rb.apply_rawboost(x, 16000, params) - x
        snr = 20 * np.log10(np.linalg.norm(x) / np.linalg.norm(noise))
        assert snr == pytest.approx(20, abs=0.01)


class TestParams:
    def test_defaults_are_valid(self):
        assert rb.RawBoostParams().validate().algo == 5

    @pytest.mark.parametrize(
        "change",
        [
            {"algo": 0},
            {"algo": 9},
            {"n_f": 0},
            {"n_bands": 0},
            {"min_coeff": 2},
            {"min_f": 9000},
            {"min_bw": 0},
            {"min_g": 1, "max_g": 0},
            {"snr_min": 50},
            {"p": 101},
        ],
    )
    def test_nonsense_is_rejected(self, change):
        with pytest.raises(ValueError):
            rb.RawBoostParams(**change).validate()

    def test_read_from_the_augment_section(self, tmp_path):
        config = configparser.ConfigParser()
        config["EXP"] = {"name": "x", "root": str(tmp_path)}
        config["AUGMENT"] = {"rawboost_algo": "3", "rawboost_snr_max": "30"}
        glob_conf.config = config
        params = rb.RawBoostParams.from_util(Util("test"))
        assert (params.algo, params.snr_max, params.snr_min) == (3, 30, 10)

    @pytest.mark.parametrize(
        "key, value",
        [
            ("rawboost_algo", "0"),
            ("rawboost_p", "many"),
            ("rawboost_algo", "4.9"),  # not silently cut to 4
            ("rawboost_g_sd", "2.5"),
            ("rawboost_p", "inf"),
            ("rawboost_snr_min", "nan"),
        ],
    )
    def test_bad_config_is_an_error(self, tmp_path, key, value):
        config = configparser.ConfigParser()
        config["EXP"] = {"name": "x", "root": str(tmp_path)}
        config["AUGMENT"] = {key: value}
        glob_conf.config = config
        with pytest.raises(NkululukoError, match="AUGMENT"):
            rb.RawBoostParams.from_util(Util("test"))


class TestRawboostSignal:
    def test_stereo_channels_get_their_own_noise(self):
        x = _signal(16000)
        stereo = np.stack([x, x])
        np.random.seed(0)
        out = rawboost_signal(stereo, 16000, rb.RawBoostParams(algo=1))
        assert out.shape == stereo.shape
        assert not np.allclose(out[0], out[1])

    def test_the_result_fits_an_audio_file(self):
        x = _signal(16000)
        out = rawboost_signal(x, 16000, rb.RawBoostParams(algo=3, snr_min=0, snr_max=0))
        assert np.amax(np.abs(out)) <= 1
        assert out.dtype == np.float32


@pytest.fixture
def setup_conf(tmp_path):
    config = configparser.ConfigParser()
    config["EXP"] = {"type": "classification", "name": "testexp", "root": str(tmp_path)}
    config["DATA"] = {"target": "emotion", "databases": "['emodb']"}
    config["MODEL"] = {"type": "xgb"}
    config["FEATS"] = {"type": "['os']"}
    config["AUGMENT"] = {"rawboost_algo": "4"}
    glob_conf.config = config
    yield config
    glob_conf.config = None


def _df(files, seconds=1):
    idx = pd.MultiIndex.from_arrays(
        [
            files,
            [pd.Timedelta(0)] * len(files),
            [pd.Timedelta(seconds=seconds)] * len(files),
        ],
        names=["file", "start", "end"],
    )
    return pd.DataFrame({"emotion": range(len(files))}, index=idx)


def _wav(path, sr=16000, channels=1):
    x = _signal(sr, seconds=1)
    audiofile.write(str(path), np.stack([x] * channels) if channels > 1 else x, sr)


class TestAugmenter:
    def test_writes_a_noisy_copy_of_every_file(self, tmp_path, setup_conf):
        files = []
        for i, sr in enumerate((8000, 16000)):
            path = tmp_path / f"f{i}.wav"
            _wav(path, sr)
            files.append(str(path))
        df = _df(files)
        out = AugmenterRawBoost(df).augment("all")
        new_files = out.index.get_level_values(0)
        assert list(out["emotion"]) == list(df["emotion"])
        for old, new, sr in zip(files, new_files, (8000, 16000)):
            assert new != old
            assert audiofile.sampling_rate(new) == sr
            assert not np.allclose(audiofile.read(new)[0], audiofile.read(old)[0])
        # start/end are kept
        assert list(out.index.get_level_values(2)) == list(df.index.get_level_values(2))

    def test_stereo_and_non_wav_inputs(self, tmp_path, setup_conf):
        stereo = tmp_path / "stereo.wav"
        _wav(stereo, channels=2)
        flac = tmp_path / "x.flac"
        _wav(flac)
        out = AugmenterRawBoost(_df([str(stereo), str(flac)])).augment("all")
        new_stereo, new_flac = out.index.get_level_values(0)
        assert audiofile.channels(new_stereo) == 2
        assert new_flac.endswith(".flac")

    def test_same_name_other_format_does_not_collide(self, tmp_path, setup_conf):
        if "MP3" not in sf.available_formats():
            pytest.skip("this libsndfile cannot write mp3")
        wav, mp3 = tmp_path / "foo.wav", tmp_path / "foo.mp3"
        _wav(wav)
        sf.write(mp3, _signal(16000, seed=1), 16000)
        out = AugmenterRawBoost(_df([str(wav), str(mp3)])).augment("all")
        new_wav, new_mp3 = out.index.get_level_values(0)
        assert new_wav != new_mp3
        assert new_mp3.endswith("foo.mp3.wav")
        assert np.isfinite(audiofile.read(new_mp3)[0]).all()
        # each copy is made from its own source
        assert not np.allclose(
            audiofile.read(new_wav)[0][:100], audiofile.read(new_mp3)[0][:100]
        )

    def test_a_file_in_several_segments_is_written_once(self, tmp_path, setup_conf):
        path = tmp_path / "f.wav"
        _wav(path)
        starts = [pd.Timedelta(0), pd.Timedelta(seconds=0.5)]
        ends = [pd.Timedelta(seconds=0.5), pd.Timedelta(seconds=1)]
        idx = pd.MultiIndex.from_arrays(
            [[str(path)] * 2, starts, ends], names=["file", "start", "end"]
        )
        out = AugmenterRawBoost(pd.DataFrame({"emotion": [0, 1]}, index=idx)).augment(
            "all"
        )
        new_files = set(out.index.get_level_values(0))
        assert len(new_files) == 1

    def test_other_settings_go_to_another_folder(self, tmp_path, setup_conf):
        path = tmp_path / "f.wav"
        _wav(path)
        first = AugmenterRawBoost(_df([str(path)])).augment("all")
        setup_conf["AUGMENT"]["rawboost_algo"] = "3"
        second = AugmenterRawBoost(_df([str(path)])).augment("all")
        assert first.index.get_level_values(0)[0] != second.index.get_level_values(0)[0]

    def test_files_are_reused_unless_no_reuse(self, tmp_path, setup_conf):
        path = tmp_path / "f.wav"
        _wav(path)
        first = AugmenterRawBoost(_df([str(path)])).augment("all")
        new = first.index.get_level_values(0)[0]
        before = audiofile.read(new)[0]
        AugmenterRawBoost(_df([str(path)])).augment("all")
        assert np.array_equal(audiofile.read(new)[0], before)
        setup_conf["DATA"]["no_reuse"] = "True"
        AugmenterRawBoost(_df([str(path)])).augment("all")
        assert not np.array_equal(audiofile.read(new)[0], before)

    def test_invalid_settings_stop_before_any_work(self, tmp_path, setup_conf):
        setup_conf["AUGMENT"]["rawboost_algo"] = "0"
        with pytest.raises(NkululukoError, match="rawboost_algo"):
            AugmenterRawBoost(_df([str(tmp_path / "f.wav")]))


def test_the_augment_module_runs_rawboost():
    import inspect

    from nkululeko import augment, experiment

    assert '"rawboost"' in inspect.getsource(augment.doit)
    assert '"rawboost"' in inspect.getsource(experiment.Experiment.augment)
