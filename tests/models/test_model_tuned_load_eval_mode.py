"""Unit tests for TunedModel.load()'s post-reload eval mode (model_tuned.py).

Regression: load()'s standard branch (`Model.from_pretrained(...)`, used for
every wav2vec2/wavlm/hubert finetune) never called `.eval()` on the reloaded
model - only the emotion2vec branch did. A freshly constructed/loaded
nn.Module defaults to train mode, so dropout stayed active during every
predict() call made against a reloaded checkpoint (e.g. the dev/test
evaluation run right after training finishes, or a later `only_test` re-eval)
- silently degrading exactly the "final" scores this reload path exists to
report. Confirmed empirically: the reloaded weights matched the true
best-dev-CCC checkpoint bit-for-bit, but the reported dev CCC was far below
that checkpoint's live-observed training-time eval score until this fix.
"""

import types
from unittest.mock import MagicMock, patch

from nkululeko.models.model_tuned import TunedModel


def make_fake_self():
    return types.SimpleNamespace(
        torch_root="/fake/torch_root",
        config="fake-config",
        set_id=lambda run, epoch: None,
    )


class TestLoadSetsEvalMode:
    def test_standard_branch_calls_eval_after_reload(self):
        fake_self = make_fake_self()
        fake_model = MagicMock()

        with patch(
            "nkululeko.models.model_tuned.Model.from_pretrained",
            return_value=fake_model,
        ):
            TunedModel.load(fake_self, run=0, epoch=30)

        assert fake_self.model is fake_model
        fake_model.eval.assert_called_once()

    def test_emotion2vec_branch_still_calls_eval(self):
        # Regression guard: the emotion2vec branch already called .eval()
        # correctly - this fix must not change or duplicate that behavior.
        fake_self = make_fake_self()
        fake_self.emotion2vec_backbone = object()
        fake_self.model = MagicMock()

        with patch("os.path.exists", return_value=True), patch(
            "torch.load", return_value={}
        ):
            TunedModel.load(fake_self, run=0, epoch=30)

        fake_self.model.eval.assert_called_once()
