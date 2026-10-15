"""Unit tests for LSTM_Reg_model (nkululeko/models/model_lstm_regression.py),
the regression variant of the recurrent model over frame sequences."""

import numpy as np
import pytest
import torch

from nkululeko.losses.loss_ccc import ConcordanceCorCoeff
from nkululeko.models.model_lstm_regression import LSTM_Reg_model
from tests.models.test_model_lstm import (
    _ConstNet,
    experiment_context,
    frames_df,
    labels_df,
)


def _model(tmp_path, measure=None, loss=None, dann_columns="[]"):
    feats = frames_df([3, 5, 2, 4])
    df = labels_df([20.0, 30.0, 40.0, 50.0], target="age", source_db=["a", "b"] * 2)
    context = experiment_context(tmp_path, "lstm_reg", "age", dann_columns)
    context.config["EXP"]["type"] = "regression"
    if measure:
        context.config["MODEL"]["measure"] = measure
    if loss:
        context.config["MODEL"]["loss"] = loss
    return LSTM_Reg_model(df, df, feats, feats, context=context)


class TestConstructor:
    @pytest.mark.parametrize("dann_columns", ["[]", "['source_db']"])
    def test_constructed_model_trains(self, tmp_path, dann_columns):
        model = _model(tmp_path, dann_columns=dann_columns)
        model.train()
        assert (model.dann_heads is not None) == (dann_columns != "[]")
        assert np.isfinite(model.loss)
        assert model.net.out.out_features == 1

    def test_labels_are_float(self, tmp_path):
        model = _model(tmp_path)
        assert model.trainloader.dataset[0][1].dtype == np.float32

    def test_unknown_loss_is_an_error(self, tmp_path):
        with pytest.raises(Exception, match="unknown loss function"):
            _model(tmp_path, loss="nonsense")


class TestEvaluate:
    @pytest.mark.parametrize("measure", ["mse", "mae", "ccc", "pcc"])
    def test_returns_a_result_for_every_row(self, tmp_path, measure):
        model = _model(tmp_path, measure=measure)
        result, targets, predictions, loss_eval = model.evaluate(model.testloader)
        assert len(targets) == len(predictions) == 4
        assert np.isfinite(result)
        assert loss_eval >= 0.0

    def test_unknown_measure_is_an_error(self, tmp_path):
        model = _model(tmp_path, measure="nonsense")
        with pytest.raises(Exception, match="unknown measure"):
            model.evaluate(model.testloader)

    def test_eval_loss_weights_batches_by_size(self, tmp_path):
        """MSE over 3 rows at batch_size 2 must equal the per-row mean,
        not the mean of the two batch means."""
        model = _model(tmp_path)
        predictions = torch.tensor([[1.0], [1.0], [10.0]])
        model.set_testdata(labels_df([0.0, 0.0, 0.0], target="age"), frames_df([2] * 3))
        model.net = _ConstNet(predictions)

        _, _, _, loss_eval = model.evaluate(model.testloader)

        assert loss_eval == pytest.approx((predictions**2).mean().item())

    def test_one_value_last_batch_keeps_eval_loss_finite(self, tmp_path):
        """With 1-ccc, CCC is undefined on the one-value last batch (3 rows
        at batch_size 2); it must not turn the evaluation loss into NaN."""
        model = _model(tmp_path, loss="1-ccc")
        model.set_testdata(labels_df([1.0, 2.0, 3.0], target="age"), frames_df([2] * 3))

        _, _, predictions, loss_eval = model.evaluate(model.testloader)

        assert isinstance(model.criterion, ConcordanceCorCoeff)
        assert np.isfinite(loss_eval)
        assert torch.isfinite(predictions).all()


class TestTrain:
    def test_non_finite_batch_is_skipped(self, tmp_path):
        """CCC is undefined on a one-value batch; that batch must neither
        update the weights nor enter the epoch loss."""
        model = _model(tmp_path, loss="1-ccc")
        model.trainloader = model.get_loader(
            labels_df([3.0], target="age"), frames_df([4]), False
        )
        before = model.net.out.weight.clone()

        model.train()

        assert model.loss == 0.0
        assert torch.equal(model.net.out.weight, before)
