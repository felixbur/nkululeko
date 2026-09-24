"""Regression follow-up to GH #438: TunedModel's inner Trainer.compute_loss
(nkululeko/models/model_tuned.py) casts logits to fp32 before the criterion
call (`logits = logits.float()`), to fix a class-weighted CrossEntropyLoss
crash under fp16 (Half logits vs. the weight tensor's Float dtype).

But TunedModel._match_loss_dtype(targets, logits, is_classifier) aligns
*regression* targets to whatever dtype `logits` has at the point it's
called - so calling it with the *original* (possibly Half, under fp16)
logits, and only afterwards casting logits to float32 for the criterion
call, passes Float logits against Half targets to MSE/L1/CCC/PCC instead:
the same class of dtype-mismatch crash the #438 fix was meant to remove,
just moved to the regression side. The fix must cast logits to fp32
*first*, then call _match_loss_dtype against that already-fp32 tensor, so
targets end up fp32 too.

This test exercises the real interaction between _match_loss_dtype and the
fp32 cast (not just isolated PyTorch dtype rules, unlike
test_model_tuned_class_weight_fp16.py), against a real MSELoss, with Half
logits reproducing the fp16-training scenario on CPU.
"""

import torch

from nkululeko.models.model_tuned import TunedModel


class TestRegressionLossUnderHalfLogits:
    def test_match_before_float_cast_leaves_targets_on_half(self):
        """Documents the exact bug: matching targets to the *original*
        (Half) logits dtype, then separately casting logits to float32
        for the criterion, leaves targets on Half while logits are Float.

        Whether this actually raises depends on the backend: CUDA kernels
        for elementwise losses generally require an exact dtype match
        (the fp16=True/GPU scenario #438 and this both apply to), while
        CPU silently type-promotes and returns a technically-valid-looking
        result anyway - so this asserts on the dtype mismatch itself
        (the actual defect) rather than relying on an exception, which
        would make the test backend-dependent and non-portable.
        """
        logits = torch.randn(4, dtype=torch.half)
        targets = torch.tensor([0.1, 0.4, 0.9, 0.2], dtype=torch.float32)

        targets = TunedModel._match_loss_dtype(targets, logits, is_classifier=False)
        logits = logits.float()

        assert targets.dtype == torch.half
        assert logits.dtype == torch.float32
        assert targets.dtype != logits.dtype

    def test_float_cast_before_match_fixes_mse(self):
        """The fixed ordering: cast logits to fp32 first, then align
        targets to that already-fp32 tensor."""
        logits = torch.randn(4, dtype=torch.half)
        targets = torch.tensor([0.1, 0.4, 0.9, 0.2], dtype=torch.float32)

        logits = logits.float()
        targets = TunedModel._match_loss_dtype(targets, logits, is_classifier=False)
        assert targets.dtype == torch.float32

        criterion = torch.nn.MSELoss()
        loss = criterion(logits, targets)
        assert torch.isfinite(loss)

    def test_classification_targets_stay_long_regardless_of_order(self):
        """Sanity check: this reordering must not affect the classification
        path at all - targets must remain Long either way."""
        logits = torch.randn(4, 2, dtype=torch.half)
        targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

        # Old order (match then float).
        matched_old = TunedModel._match_loss_dtype(targets, logits, is_classifier=True)
        assert matched_old.dtype == torch.long

        # New order (float then match).
        matched_new = TunedModel._match_loss_dtype(
            targets, logits.float(), is_classifier=True
        )
        assert matched_new.dtype == torch.long
