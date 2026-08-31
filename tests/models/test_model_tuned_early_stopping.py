"""Unit tests for TunedModel._build_callbacks (nkululeko/models/model_tuned.py).

Regression: finetuning's TrainingArguments set load_best_model_at_end=True
but registered no EarlyStoppingCallback, so training always ran for the full
configured epoch count regardless of dev performance - load_best_model_at_end
only picks the best already-saved checkpoint afterward, it does not stop
training early the way MODEL.patience does for every other model type
(SVM/MLP/CNN via modelrunner.do_epochs()'s own patience loop).
"""

import types

import transformers

from nkululeko.models.model_tuned import TunedModel


class DummyUtil:
    def __init__(self, patience):
        self.patience = patience
        self.debug_messages = []

    def config_val(self, section, key, default):
        if section == "MODEL" and key == "patience":
            return self.patience
        return default

    def debug(self, message):
        self.debug_messages.append(message)


def build(patience, evals_per_epoch=5):
    fake_self = types.SimpleNamespace(util=DummyUtil(patience))
    return TunedModel._build_callbacks(fake_self, evals_per_epoch), fake_self.util


class TestBuildCallbacks:
    def test_no_patience_only_tensorboard_callback(self):
        callbacks, util = build(False)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], transformers.integrations.TensorBoardCallback)
        assert util.debug_messages == []

    def test_patience_adds_scaled_early_stopping_callback(self):
        callbacks, util = build("2", evals_per_epoch=5)
        assert len(callbacks) == 2
        early_stop = callbacks[1]
        assert isinstance(early_stop, transformers.EarlyStoppingCallback)
        assert early_stop.early_stopping_patience == 10
        assert len(util.debug_messages) == 1

    def test_scaling_uses_the_given_evals_per_epoch(self):
        callbacks, _ = build("3", evals_per_epoch=4)
        assert callbacks[1].early_stopping_patience == 12
