"""Tests for frame sequences (nkululeko/utils/sequences.py) and how the
pipeline handles them: alignment, storage, import, pooling, opensmile
frame extraction and scaling (GH #471)."""

import configparser

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from nkululeko.experiment_context import ExperimentContext
from nkululeko.utils import sequences as seq
from nkululeko.utils.util import Util


def td(seconds):
    return pd.Timedelta(seconds=seconds)


SEG_A = ("/data/x/a.wav", td(0), td(1))
SEG_B = ("/data/x/b.wav", td(0.5), pd.NaT)


def frames(segment, values, step=0.01):
    """One feature column 'f' and 'g' = 2 * f, one row per value."""
    index = seq.sequence_index(segment, np.arange(len(values)) * step)
    values = np.asarray(values, dtype=float)
    return pd.DataFrame({"f": values, "g": 2 * values}, index=index)


def make_util(tmp_path, feats=None, model_type="svm"):
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "EXP": {"root": str(tmp_path), "name": "t"},
            "DATA": {"target": "emotion"},
            "FEATS": feats or {"type": "['os']"},
            "MODEL": {"type": model_type, "n_jobs": "0"},
        }
    )
    return Util("test", context=ExperimentContext(config=config))


class TestHelpers:
    def test_times_are_in_nanoseconds(self):
        index = seq.sequence_index(SEG_A, [0, 0.01])
        for level in ("start", "end", "frame"):
            assert index.get_level_values(level).dtype == "timedelta64[ns]"

    def test_sequence_index_is_relative_to_the_segment(self):
        index = seq.sequence_index(SEG_B, [0, 0.01])
        assert list(index.names) == ["file", "start", "end", "frame"]
        assert index[0][1] == td(0.5)
        assert list(index.get_level_values("frame")) == [td(0), td(0.01)]

    def test_is_sequence_and_n_segments(self):
        df = pd.concat([frames(SEG_A, [1, 2, 3]), frames(SEG_B, [4, 5])])
        assert seq.is_sequence(df)
        assert seq.n_segments(df) == 2
        assert not seq.is_sequence(seq.pool(df))

    def test_pool_one_row_per_segment_in_order(self):
        df = pd.concat([frames(SEG_B, [4, 6]), frames(SEG_A, [1, 2, 3])])
        pooled = seq.pool(df, ["mean", "std", "max"])
        assert [seg[0] for seg in pooled.index] == [SEG_B[0], SEG_A[0]]
        assert list(pooled.columns) == [
            "f_mean",
            "g_mean",
            "f_std",
            "g_std",
            "f_max",
            "g_max",
        ]
        assert pooled["f_mean"].iloc[0] == 5
        assert pooled["f_std"].iloc[1] == pytest.approx(np.std([1, 2, 3]))

    def test_pool_leaves_vectors_alone(self):
        df = pd.DataFrame({"f": [1.0]}, index=pd.MultiIndex.from_tuples([SEG_A]))
        assert seq.pool(df) is df

    def test_pool_unknown_statistic(self):
        with pytest.raises(ValueError, match="unknown pooling"):
            seq.pool(frames(SEG_A, [1, 2]), ["mode"])

    def test_to_arrays_orders_frames_by_time(self):
        df = frames(SEG_A, [1, 2, 3]).iloc[[2, 0, 1]]
        index, arrays = seq.to_arrays(df)
        assert list(index) == [SEG_A]
        np.testing.assert_array_equal(arrays[0][:, 0], [1, 2, 3])

    def test_align_frame_labels_at_another_rate(self):
        feats = frames(SEG_A, [0, 0, 0, 0], step=0.01)  # 0, 10, 20, 30 ms
        labels = pd.DataFrame(
            {"arousal": [0.1, 0.5]},
            index=seq.sequence_index(SEG_A, [0.005, 0.02]),
        )
        aligned = seq.align_frame_labels(labels, feats)
        assert aligned.index.equals(feats.index)
        np.testing.assert_allclose(aligned["arousal"], [np.nan, 0.1, 0.5, 0.5])

    def test_csv_round_trip_in_seconds(self, tmp_path):
        df = pd.concat([frames(SEG_A, [1, 2]), frames(SEG_B, [3])])
        path = tmp_path / "frames.csv"
        seq.write_sequence_csv(df, path)
        written = pd.read_csv(path)
        assert list(written.columns[:4]) == ["file", "start", "end", "frame"]
        assert written["frame"].tolist() == [0.0, 0.01, 0.0]
        back = seq.read_sequence_csv(path)
        pd.testing.assert_frame_equal(back, df, check_freq=False)

    def test_pooling_stats_from_config(self, tmp_path):
        util = make_util(tmp_path)
        assert seq.pooling_stats(util) == ["mean", "std"]
        util.config["FEATS"]["pooling"] = "['max']"
        assert seq.pooling_stats(util) == ["max"]


class TestFilterFilepath:
    def test_labels_select_all_frames_of_their_segments(self, tmp_path):
        util = make_util(tmp_path)
        feats = pd.concat([frames(SEG_A, [1, 2, 3]), frames(SEG_B, [4, 5])])
        labels = pd.DataFrame({"y": [1]}, index=pd.MultiIndex.from_tuples([SEG_B]))
        kept = util.filter_filepath(labels, feats)
        assert len(kept) == 2
        assert set(kept["f"]) == {4, 5}

    def test_frames_select_each_label_once(self, tmp_path):
        util = make_util(tmp_path)
        feats = pd.concat([frames(SEG_B, [4, 5]), frames(SEG_A, [1, 2, 3])])
        labels = pd.DataFrame(
            {"y": [1, 2]}, index=pd.MultiIndex.from_tuples([SEG_A, SEG_B])
        )
        kept = util.filter_filepath(feats, labels)
        assert kept["y"].tolist() == [2, 1]  # in the frames' segment order


class TestStorage:
    def test_csv_store_keeps_the_frame_level(self, tmp_path):
        util = make_util(tmp_path)
        df = frames(SEG_A, [1, 2])
        path = str(tmp_path / "store.csv")
        util.write_store(df, path, "csv")
        back = util.get_store(path, "csv")
        assert seq.is_sequence(back)
        assert len(back) == 2


def _write_wav(path, seconds, sr=16000):
    rng = np.random.default_rng(0)
    sf.write(path, rng.uniform(-0.1, 0.1, int(seconds * sr)).astype(np.float32), sr)


def _data_df(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_wav(a, 1.0)
    _write_wav(b, 1.0)
    index = pd.MultiIndex.from_tuples(
        [(str(a), td(0), td(0.5)), (str(b), td(0.2), td(0.9))],
        names=["file", "start", "end"],
    )
    return pd.DataFrame({"emotion": ["x", "y"]}, index=index)


class TestFeatureExtractor:
    def _extract(self, tmp_path, model_type, pooling=None):
        from nkululeko.feature_extractor import FeatureExtractor

        feats = {"type": "['os']", "level": "frame", "no_reuse": "True"}
        if pooling:
            feats["pooling"] = pooling
        util = make_util(tmp_path, feats, model_type)
        data_df = _data_df(tmp_path)
        extractor = FeatureExtractor(
            data_df, ["os"], "test", "all", context=util.context
        )
        return extractor, extractor.extract(), data_df

    def test_sequence_model_gets_frames_relative_to_segment(self, tmp_path):
        _, feats, data_df = self._extract(tmp_path, "lstm")
        assert seq.is_sequence(feats)
        assert list(seq.segment_index(feats)) == list(data_df.index)
        first = feats.xs(data_df.index[1], level=[0, 1, 2], drop_level=False)
        assert first.index.get_level_values("frame")[0] == td(0)
        assert feats.shape[1] == 25  # eGeMAPSv02 LLDs

    def test_vector_model_gets_pooled_frames(self, tmp_path):
        _, feats, data_df = self._extract(tmp_path, "svm", "['mean', 'max']")
        assert not seq.is_sequence(feats)
        assert list(feats.index) == list(data_df.index)
        assert feats.shape[1] == 50
        assert feats.columns[0].endswith("_mean")

    def test_extract_sample_pools_like_extract(self, tmp_path):
        extractor, feats, data_df = self._extract(tmp_path, "svm")
        signal, sr = sf.read(data_df.index[0][0])
        sample = extractor.extract_sample(signal[: sr // 2], sr)
        assert sample.shape == (1, feats.shape[1])
        np.testing.assert_allclose(sample[0], feats.iloc[0].to_numpy(), rtol=1e-4)

    def test_mixing_frames_and_vectors_for_a_sequence_model(self, tmp_path):
        from nkululeko.feature_extractor import FeatureExtractor

        util = make_util(tmp_path, model_type="lstm")
        extractor = FeatureExtractor(
            _data_df(tmp_path), [], "test", "all", context=util.context
        )
        vectors = pd.DataFrame({"v": [1.0]}, index=pd.MultiIndex.from_tuples([SEG_A]))
        with pytest.raises(Exception, match="mixes frame sequences"):
            extractor._join([frames(SEG_A, [1, 2]), vectors])


class TestImport:
    def test_frame_csv_is_imported_as_sequences(self, tmp_path):
        from nkululeko.feat_extract.feats_import import ImportSet
        from nkululeko.experiment_context import use_context

        data_df = _data_df(tmp_path)
        seg = data_df.index[0]
        df = frames(seg, [1, 2, 3])
        other = frames(("/elsewhere.wav", td(0), td(1)), [9])
        path = tmp_path / "imported.csv"
        seq.write_sequence_csv(pd.concat([df, other]), path)
        util = make_util(tmp_path, {"type": "['import']", "import_file": str(path)})

        with use_context(util.context):
            importer = ImportSet("imp", data_df, "import")
        importer.extract()

        assert seq.is_sequence(importer.df)
        assert len(importer.df) == 3  # the frames of the known segment only


class TestScaling:
    def test_speaker_scaling_scales_all_frames_of_a_speaker(self, tmp_path):
        from nkululeko.scaler import Scaler

        util = make_util(tmp_path)
        feats = pd.concat([frames(SEG_A, [1, 2, 3]), frames(SEG_B, [10, 20])])
        labels = pd.DataFrame(
            {"emotion": ["x", "y"], "speaker": ["s1", "s2"]},
            index=pd.MultiIndex.from_tuples(
                [SEG_A, SEG_B], names=["file", "start", "end"]
            ),
        )
        scaler = Scaler(
            labels, None, feats.copy(), None, "speaker", context=util.context
        )
        scaled = scaler.speaker_scale_df(labels, feats.copy())
        for segment in (SEG_A, SEG_B):
            rows = scaled.index.droplevel("frame").isin([segment])
            assert scaled.loc[rows, "f"].mean() == pytest.approx(0.0)
            assert scaled.loc[rows, "f"].std(ddof=0) == pytest.approx(1.0)


class TestSSLFrames:
    """wav2vec2/HuBERT/WavLM frame output, with a stand-in model."""

    class _Model:
        class config:
            conv_stride = [5, 2, 2, 2, 2, 2, 2]

        def __call__(self, y):
            # (batch, frames, dims): one frame per 320 samples, dims 4
            import torch

            n = y.shape[1] // 320
            return (torch.arange(n * 4, dtype=torch.float32).reshape(1, n, 4),)

    def _extractor(self, tmp_path, level):
        from nkululeko.experiment_context import use_context
        from nkululeko.feat_extract.feats_wav2vec2 import Wav2vec2

        util = make_util(tmp_path, {"type": "['wav2vec2']", "level": level})
        with use_context(util.context):
            extractor = Wav2vec2("w2v", _data_df(tmp_path), "wav2vec2")
        extractor.model = self._Model()
        extractor.processor = lambda signal, sampling_rate: {
            "input_values": np.asarray(signal)
        }
        extractor.model_initialized = True
        return extractor

    def test_frame_step_from_conv_strides(self, tmp_path):
        assert self._extractor(tmp_path, "frame")._ssl_frame_step() == 0.02

    def test_embeddings_per_frame_or_pooled(self, tmp_path):
        extractor = self._extractor(tmp_path, "frame")
        signal = np.zeros((1, 3200), dtype=np.float32)
        frames = extractor.get_embeddings(signal, 16000, "f", pooled=False)
        pooled = extractor.get_embeddings(signal, 16000, "f")
        assert frames.shape == (10, 4)
        np.testing.assert_allclose(pooled, frames.mean(axis=0))

    def test_extract_gives_frames_relative_to_segment(self, tmp_path, monkeypatch):
        import torch

        import nkululeko.feat_extract.feats_wav2vec2 as module

        def load(file, frame_offset, num_frames, normalize=True):
            # torchaudio.load needs the torchcodec backend in newer
            # versions; the audio backend is not what this test is about
            data, sr = sf.read(
                file, start=frame_offset, frames=num_frames, dtype="float32"
            )
            return torch.from_numpy(data).reshape(1, -1), sr

        monkeypatch.setattr(module.torchaudio, "load", load)
        extractor = self._extractor(tmp_path, "frame")
        extractor.extract()
        df = extractor.df
        assert seq.is_sequence(df)
        assert seq.n_segments(df) == 2
        # segments of 0.5 s and 0.7 s at 20 ms per frame
        sizes = [len(a) for a in seq.to_arrays(df)[1]]
        assert sizes == [25, 35]
        assert df.index.get_level_values("frame")[1] == td(0.02)
        assert df.dtypes.iloc[0] == np.float32

    def test_supports_frames(self):
        from nkululeko.feat_extract.feats_hubert import Hubert
        from nkululeko.feat_extract.feats_wav2vec2 import Wav2vec2
        from nkululeko.feat_extract.feats_wavlm import Wavlm

        assert Wav2vec2.supports_frames and Hubert.supports_frames
        assert Wavlm.supports_frames
