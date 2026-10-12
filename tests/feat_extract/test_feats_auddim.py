"""Tests for nkululeko/feat_extract/feats_auddim.py — AuddimSet.

GH #459 item 5(a): onnxruntime silently falls back to CPU when CUDA was
requested but isn't actually available (e.g. missing cuDNN 9).
"""

from unittest.mock import MagicMock, patch

from nkululeko.feat_extract.feats_auddim import AuddimSet


class _FakeOnnxModel:
    def __init__(self, providers=("CPUExecutionProvider",)):
        self.sess = MagicMock(get_providers=MagicMock(return_value=list(providers)))


class TestGpuFallbackWarning:
    def _load(self, providers, device):
        instance = AuddimSet.__new__(AuddimSet)
        instance.util = MagicMock()
        instance.util.config_val.side_effect = lambda section, key, default: {
            ("MODEL", "device"): device,
        }.get((section, key), default)
        fake_model = _FakeOnnxModel(providers=providers)
        with (
            patch("os.path.isdir", return_value=True),
            patch(
                "nkululeko.feat_extract.feats_auddim.audonnx.load",
                return_value=fake_model,
            ),
        ):
            instance._load_model()
        return instance

    def test_warns_when_cuda_requested_but_fell_back_to_cpu(self):
        instance = self._load(("CPUExecutionProvider",), "cuda")

        assert instance.util.warn.called
        assert "CUDA" in instance.util.warn.call_args[0][0]

    def test_no_warning_when_cuda_is_actually_active(self):
        instance = self._load(
            ("CUDAExecutionProvider", "CPUExecutionProvider"), "cuda"
        )

        assert not instance.util.warn.called

    def test_no_warning_when_cpu_was_requested(self):
        instance = self._load(("CPUExecutionProvider",), "cpu")

        assert not instance.util.warn.called
