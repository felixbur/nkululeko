"""Unit tests for the recurrent models over frame sequences
(nkululeko/models/rnn_core.py, model_lstm.py).

The models take frame sequences from the feature pipeline (index
(file, start, end, frame)), so the tests build small synthetic frame
tables instead of audio.
"""

import configparser

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from nkululeko.experiment_context import ExperimentContext
from nkululeko.models.model_lstm import LSTMModel
from nkululeko.models.rnn_core import RNNNet, SequenceDataset, collate
from nkululeko.utils.sequences import sequence_index


def _segments(n):
    return [
        (f"/data/s{i}.wav", pd.Timedelta(0), pd.Timedelta(seconds=1)) for i in range(n)
    ]


def frames_df(lengths, n_feats=3, seed=0):
    """Frame sequences with the given number of frames per segment."""
    rng = np.random.default_rng(seed)
    parts = []
    for segment, n in zip(_segments(len(lengths)), lengths):
        index = sequence_index(segment, np.arange(n) * 0.01)
        parts.append(
            pd.DataFrame(
                rng.normal(size=(n, n_feats)),
                index=index,
                columns=[f"f{i}" for i in range(n_feats)],
            )
        )
    return pd.concat(parts)


def labels_df(values, target="emotion", **columns):
    index = pd.MultiIndex.from_tuples(
        _segments(len(values)), names=["file", "start", "end"]
    )
    return pd.DataFrame({target: values, **columns}, index=index)


def experiment_context(tmp_path, model_type, target, dann_columns="[]", labels=None):
    config = configparser.ConfigParser()
    config.read_dict(
        {
            "EXP": {"root": str(tmp_path), "name": "t"},
            "DATA": {"target": target},
            "FEATS": {"type": "['os']", "level": "frame"},
            "MODEL": {
                "type": model_type,
                "device": "cpu",
                "n_jobs": "0",
                "dann_columns": dann_columns,
            },
            "RNN": {"hidden_size": "4", "num_layers": "1", "batch_size": "2"},
        }
    )
    return ExperimentContext(config=config, labels=labels)


class TestSequenceDataset:
    def test_items_follow_the_label_rows_and_keep_their_length(self):
        feats = frames_df([3, 5, 2])
        # labels in another order than the features
        df = labels_df([0, 1, 2]).iloc[[2, 0, 1]]

        dataset = SequenceDataset(df, feats, "emotion", np.int64)

        assert [len(dataset[i][0]) for i in range(3)] == [2, 3, 5]
        assert [int(dataset[i][1]) for i in range(3)] == [2, 0, 1]

    def test_missing_frames_are_an_error(self):
        feats = frames_df([3, 5])
        with pytest.raises(ValueError, match="no feature frames"):
            SequenceDataset(labels_df([0, 1, 2]), feats, "emotion", np.int64)

    def test_domain_labels_add_a_third_item(self):
        feats = frames_df([3, 5])
        dom = np.array([[0], [1]])
        dataset = SequenceDataset(labels_df([0, 1]), feats, "emotion", np.int64, dom)
        assert len(dataset[1]) == 3
        assert int(dataset[1][2][0]) == 1


class TestCollate:
    def test_pads_to_the_longest_sequence(self):
        batch = [
            (torch.ones(2, 3), np.int64(0)),
            (torch.ones(4, 3), np.int64(1)),
        ]
        frames, lengths, labels = collate(batch)
        assert frames.shape == (2, 4, 3)
        assert lengths.tolist() == [2, 4]
        assert frames[0, 2:].abs().sum() == 0
        assert labels.tolist() == [0, 1]


class TestRNNNet:
    def _net(self, bidirectional=False):
        torch.manual_seed(0)
        return RNNNet(3, 4, 2, 5, bidirectional, 0.0).eval()

    def test_output_shapes(self):
        net = self._net()
        out, pooled = net(torch.randn(2, 6, 3), torch.tensor([6, 4]), True)
        assert out.shape == (2, 5)
        assert pooled.shape == (2, 4)

    @pytest.mark.parametrize("bidirectional", [False, True])
    def test_padding_does_not_change_the_output(self, bidirectional):
        net = self._net(bidirectional)
        seq = torch.randn(1, 3, 3)
        alone = net(seq, torch.tensor([3]))
        padded = torch.cat([seq, torch.full((1, 4, 3), 9.0)], dim=1)
        in_batch = net(padded, torch.tensor([3]))
        torch.testing.assert_close(alone, in_batch)

    def test_bidirectional_doubles_feat_dim(self):
        assert self._net(bidirectional=True).feat_dim == 8


class TestConstructor:
    """Runs the real __init__."""

    @pytest.mark.parametrize("dann_columns", ["[]", "['source_db']"])
    def test_constructed_model_trains_and_predicts(self, tmp_path, dann_columns):
        feats = frames_df([3, 5, 2, 4])
        df = labels_df([0, 1, 2, 0], source_db=["a", "b", "a", "b"])
        context = experiment_context(
            tmp_path, "lstm", "emotion", dann_columns, labels=["a", "b", "c"]
        )

        model = LSTMModel(df, df, feats, feats, context=context)
        model.train()
        uar, targets, predictions, logits, loss_eval = model.evaluate(model.testloader)

        assert (model.dann_heads is not None) == (dann_columns != "[]")
        assert np.isfinite(model.loss)
        assert logits.shape == (4, 3)
        assert targets.tolist() == [0, 1, 2, 0]
        assert np.isfinite(loss_eval)

    def test_one_vector_per_sample_is_an_error(self, tmp_path):
        df = labels_df([0, 1])
        feats = pd.DataFrame({"f0": [0.1, 0.2]}, index=df.index)
        context = experiment_context(tmp_path, "lstm", "emotion", labels=["a", "b"])
        with pytest.raises(Exception, match="FEATS.level = frame"):
            LSTMModel(df, df, feats, feats, context=context)


@pytest.fixture
def lstm_model(tmp_path):
    feats = frames_df([3, 5, 2])
    df = labels_df([1, 0, 2])
    context = experiment_context(tmp_path, "lstm", "emotion", labels=["a", "b", "c"])
    return LSTMModel(df, df, feats, feats, context=context)


class _ConstNet(nn.Module):
    """Returns fixed outputs per row, in order."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.pos = 0

    def forward(self, x, lengths, return_features=False):
        out = self.outputs[self.pos : self.pos + len(x)]
        self.pos += len(x)
        return out


class TestEvaluate:
    def test_eval_loss_weights_batches_by_size(self, lstm_model):
        """3 rows at batch_size 2: the loss must be the per-row mean, not
        the mean of the two batch means."""
        logits = torch.tensor([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        lstm_model.df_test = labels_df([0, 0, 0])
        lstm_model.set_testdata(lstm_model.df_test, frames_df([2, 2, 2]))
        lstm_model.net = _ConstNet(logits)

        _, _, _, _, loss_eval = lstm_model.evaluate(lstm_model.testloader)

        per_row = nn.functional.cross_entropy(
            logits, torch.tensor([0, 0, 0]), reduction="none"
        )
        assert loss_eval == pytest.approx(per_row.mean().item(), rel=1e-5)


class TestProbas:
    def test_rows_sum_to_one_and_follow_df_test(self, lstm_model):
        logits = torch.tensor([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]])
        probas = lstm_model.get_probas(logits)
        assert list(probas.index) == list(lstm_model.df_test.index)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)

    def test_keeps_classes_missing_from_test(self, lstm_model):
        lstm_model.df_test = labels_df([1, 2])
        logits = torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])

        probas = lstm_model.get_probas(logits)

        assert list(probas.columns) == [0, 1, 2]
        np.testing.assert_allclose(
            probas[2].to_numpy(), torch.softmax(logits, dim=1)[:, 2].numpy()
        )


class TestStoreLoad:
    def test_load_restores_stored_weights(self, lstm_model, tmp_path):
        lstm_model.store_path = str(tmp_path / "m.model")
        lstm_model.set_id = lambda run, epoch: None
        stored = {k: v.clone() for k, v in lstm_model.net.state_dict().items()}
        lstm_model.store()
        with torch.no_grad():
            for p in lstm_model.net.parameters():
                p.add_(1.0)

        lstm_model.load(run=0, epoch=0)

        for k, v in lstm_model.net.state_dict().items():
            assert torch.equal(v, stored[k])
