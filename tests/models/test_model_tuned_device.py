"""Unit tests for TunedModel._resolve_device (nkululeko/models/model_tuned.py).

CUDA_VISIBLE_DEVICES only ever accepts physical GPU indices, never a torch
device string like "cuda" or "cuda:0" - setting it to those hides every GPU
instead of selecting one. These tests lock in that only a numeric index (or
"cuda:<idx>") is turned into a CUDA_VISIBLE_DEVICES value, while the
torch-facing device string returned is always plain "cpu" or "cuda".
"""

import types

from nkululeko.models.model_tuned import TunedModel


class DummyUtil:
    def __init__(self):
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)


def resolve(device):
    fake_self = types.SimpleNamespace(util=DummyUtil())
    return TunedModel._resolve_device(fake_self, device), fake_self.util.warnings


class TestResolveDevice:
    def test_cpu_passthrough(self):
        (torch_device, cuda_visible), warnings = resolve("cpu")
        assert torch_device == "cpu"
        assert cuda_visible is None
        assert warnings == []

    def test_bare_cuda_does_not_set_cuda_visible_devices(self):
        (torch_device, cuda_visible), warnings = resolve("cuda")
        assert torch_device == "cuda"
        assert cuda_visible is None
        assert warnings == []

    def test_numeric_index(self):
        (torch_device, cuda_visible), warnings = resolve("4")
        assert torch_device == "cuda"
        assert cuda_visible == "4"
        assert warnings == []

    def test_comma_separated_indices(self):
        (torch_device, cuda_visible), warnings = resolve("0,1")
        assert torch_device == "cuda"
        assert cuda_visible == "0,1"
        assert warnings == []

    def test_cuda_colon_index_extracts_index(self):
        (torch_device, cuda_visible), warnings = resolve("cuda:0")
        assert torch_device == "cuda"
        assert cuda_visible == "0"
        assert warnings == []

    def test_unrecognized_value_falls_back_to_cuda_and_warns(self):
        (torch_device, cuda_visible), warnings = resolve("gpu7")
        assert torch_device == "cuda"
        assert cuda_visible is None
        assert len(warnings) == 1
