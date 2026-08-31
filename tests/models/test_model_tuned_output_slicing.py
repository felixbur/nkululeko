"""Unit tests for ModelOutput/ModelOutputReg.__getitem__ slicing (model_tuned.py).

Regression: HF's Trainer.prediction_step() calls
`loss, outputs = self.compute_loss(model, inputs, return_outputs=True)` (our
custom Trainer override), then does `logits = outputs[1:]` - assuming index
0 is a loss value to skip, which is the convention for models that compute
their own loss internally. Our Model.forward() never does that: index 0
(logits) is the only real output, and every other field defaults to None.

ModelOutput (classification) already had a fallback for this: if slicing
filters out everything, fall back to `(self.logits,)`. ModelOutputReg
(regression) was missing that fallback entirely, so `outputs[1:]` came back
as an empty tuple - HF's EvalPrediction.predictions ended up `()`, and
compute_metrics raised `ValueError: Empty predictions tuple received: ()`
on every regression finetuning run, only surfacing once a real dev
evaluation was actually exercised.
"""

import torch

from nkululeko.models.model_tuned import ModelOutput, ModelOutputReg


class TestModelOutputSlicing:
    def test_classification_slice_from_one_falls_back_to_logits(self):
        logits = torch.tensor([1.0, 2.0])
        output = ModelOutput(logits=logits)
        assert output[1:] == (logits,)

    def test_regression_slice_from_one_falls_back_to_logits(self):
        logits = torch.tensor([0.5])
        output = ModelOutputReg(logits=logits)
        assert output[1:] == (logits,)

    def test_regression_slice_includes_non_none_fields(self):
        logits = torch.tensor([0.5])
        hidden = torch.tensor([1.0, 2.0])
        output = ModelOutputReg(logits=logits, hidden_states=hidden)
        assert output[1:] == (hidden,)

    def test_regression_full_slice_returns_all_non_none_fields(self):
        logits = torch.tensor([0.5])
        output = ModelOutputReg(logits=logits)
        assert output[:] == (logits,)
