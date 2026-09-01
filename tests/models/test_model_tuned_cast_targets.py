"""Unit tests for TunedModel._cast_targets (nkululeko/models/model_tuned.py).

Regression: the HF Trainer's compute_loss override unconditionally did
`targets.type(torch.long)`, correct for classification but silently
truncating every continuous regression target to an integer (e.g. a
grbas_score of 0.6666 became 0). This corrupted the training signal for
every regression loss (1-ccc, 1-pcc, mse, mae) - not a crash, just near-total
loss of information - and for plain torch losses like MSELoss (which don't
internally re-cast, unlike our own CCC/PCC classes) it crashed outright with
a dtype mismatch during the backward pass under fp16.
"""

import torch

from nkululeko.models.model_tuned import TunedModel


class TestCastTargets:
    def test_classification_casts_to_long(self):
        targets = torch.tensor([0.0, 1.0, 2.0])
        result = TunedModel._cast_targets(targets, is_classifier=True)
        assert result.dtype == torch.long
        assert result.tolist() == [0, 1, 2]

    def test_regression_stays_float_and_keeps_fractional_value(self):
        targets = torch.tensor([0.6666, 1.3333, 2.9999])
        result = TunedModel._cast_targets(targets, is_classifier=False)
        assert result.dtype == torch.float32
        assert torch.allclose(result, targets, atol=1e-4)

    def test_regression_does_not_truncate_small_scores_to_zero(self):
        # The exact failure mode: grbas_score values are typically in
        # [0, ~3], so unconditional long-casting turned most targets into 0.
        targets = torch.tensor([0.1, 0.4, 0.9])
        result = TunedModel._cast_targets(targets, is_classifier=False)
        assert not torch.all(result == 0)


class TestMatchLossDtype:
    def test_regression_matches_half_precision_logits(self):
        # Regression: under fp16 training, logits come out of the model as
        # Half while _cast_targets produces float32, and torch losses like
        # MSELoss require an exact dtype match - crashing with
        # "Found dtype Float but expected Half" during the backward pass.
        targets = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float32)
        logits = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float16)
        result = TunedModel._match_loss_dtype(targets, logits, is_classifier=False)
        assert result.dtype == torch.float16

    def test_regression_matches_float32_logits(self):
        targets = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float32)
        logits = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32)
        result = TunedModel._match_loss_dtype(targets, logits, is_classifier=False)
        assert result.dtype == torch.float32

    def test_classification_targets_left_untouched(self):
        # CrossEntropyLoss accepts Long targets regardless of the logits'
        # dtype, so classification must not be coerced to match logits.
        targets = torch.tensor([0, 1, 2], dtype=torch.long)
        logits = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float16)
        result = TunedModel._match_loss_dtype(targets, logits, is_classifier=True)
        assert result.dtype == torch.long
