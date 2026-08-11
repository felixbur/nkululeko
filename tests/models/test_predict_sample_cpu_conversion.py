"""Regression test: predict_sample() must move GPU tensors to host memory
before calling .numpy().

With MODEL.device=cuda, `self.model(...)` returns a CUDA tensor. Calling
`.numpy()` on it directly raises:
    RuntimeError: can't convert cuda:0 device type tensor to numpy. Use
    Tensor.cpu() to copy the tensor to host memory first.

`MLP_Reg_model.predict_sample()` and `CNNModel.predict_sample()` did exactly
that -- `a = logits.numpy()` with no `.cpu()` first -- so every row of a
GPU-backed `nkululeko.predict --type model` run hit this error. (For
comparison, `MLPModel.predict_sample()` already guards this correctly with
`if isinstance(logits, torch.Tensor): logits = logits.cpu()`.)

No GPU is required to catch this: a fake tensor stands in for a CUDA tensor,
raising the same RuntimeError from `.numpy()` unless `.cpu()` was called
first.
"""

import numpy as np
import pytest
import torch

from nkululeko.models.model_cnn import CNNModel
from nkululeko.models.model_mlp_regression import MLP_Reg_model


class FakeCudaTensor:
    """Stands in for a torch.Tensor living on a CUDA device.

    `.numpy()` raises exactly like a real CUDA tensor would; only after
    `.cpu()` is called does `.numpy()` succeed.
    """

    def __init__(self, tensor):
        self._tensor = tensor

    def cpu(self):
        return self._tensor

    def numpy(self):
        raise RuntimeError(
            "can't convert cuda:0 device type tensor to numpy. Use "
            "Tensor.cpu() to copy the tensor to host memory first."
        )

    def reshape(self, *shape):
        return FakeCudaTensor(self._tensor.reshape(*shape))


def test_mlp_reg_model_predict_sample_converts_gpu_tensor(monkeypatch):
    model = MLP_Reg_model.__new__(MLP_Reg_model)
    # `self.device` only drives `features.to(self.device)`; the real GPU
    # environment being reproduced here is `self.model(...)` returning a
    # tensor that hasn't been copied to host memory yet -- exactly what
    # FakeCudaTensor simulates below. Using "cpu" avoids requiring real
    # CUDA hardware for this test while still exercising the exact code
    # path that used to crash: `logits.numpy()` without `.cpu()` first.
    model.device = "cpu"
    model.model = lambda x: FakeCudaTensor(torch.tensor([[0.42]]))

    result = model.predict_sample(np.array([0.1, 0.2, 0.3], dtype=np.float32))

    assert result == pytest.approx(0.42)


def test_cnn_model_predict_sample_converts_gpu_tensor(monkeypatch):
    model = CNNModel.__new__(CNNModel)
    # `self.device` only drives `features.to(self.device)`; the real GPU
    # environment being reproduced here is `self.model(...)` returning a
    # tensor that hasn't been copied to host memory yet -- exactly what
    # FakeCudaTensor simulates below. Using "cpu" avoids requiring real
    # CUDA hardware for this test while still exercising the exact code
    # path that used to crash: `logits.numpy()` without `.cpu()` first.
    model.device = "cpu"
    model.model = lambda x: FakeCudaTensor(torch.tensor([[0.1, 0.9]]))

    result = model.predict_sample(np.array([0.1, 0.2, 0.3], dtype=np.float32))

    assert result == {0: pytest.approx(0.1), 1: pytest.approx(0.9)}
