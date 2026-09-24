"""Unit tests for TunedModel._build_callbacks (nkululeko/models/model_tuned.py).

Regression: finetuning's TrainingArguments set load_best_model_at_end=True
but registered no EarlyStoppingCallback, so training always ran for the full
configured epoch count regardless of dev performance - load_best_model_at_end
only picks the best already-saved checkpoint afterward, it does not stop
training early the way MODEL.patience does for every other model type
(SVM/MLP/CNN via modelrunner.do_epochs()'s own patience loop).

Also covers GH #439: transformers.integrations.TensorBoardCallback() raises
RuntimeError at construction time (not lazily, on first use) if tensorboard
isn't installed - _build_callbacks() used to construct it unconditionally,
crashing every finetune run on a fresh install with "TensorBoardCallback
requires tensorboard to be installed" before training even started.
nkululeko has its own separate reporting/plotting and never reads what this
callback writes, so it's now skipped (with a debug message) rather than
declared as a hard dependency, when unavailable.
"""

import types
from unittest.mock import patch

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


class TestBuildCallbacksTensorboardUnavailable:
    """Patches TensorBoardCallback itself to raise RuntimeError, exactly as
    it does at construction time when tensorboard isn't installed (verified
    directly against the real transformers source: TensorBoardCallback.
    __init__ raises RuntimeError("TensorBoardCallback requires tensorboard
    to be installed...") before doing anything else) - so this reproduces
    the actual reported crash, not just a stand-in for it."""

    def test_skips_tensorboard_callback_without_crashing(self):
        with patch(
            "transformers.integrations.TensorBoardCallback",
            side_effect=RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed."
            ),
        ):
            callbacks, util = build(False)

        assert callbacks == []
        assert any("tensorboard" in msg for msg in util.debug_messages)

    def test_early_stopping_still_added_without_tensorboard(self):
        with patch(
            "transformers.integrations.TensorBoardCallback",
            side_effect=RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed."
            ),
        ):
            callbacks, _ = build("2", evals_per_epoch=5)

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], transformers.EarlyStoppingCallback)
