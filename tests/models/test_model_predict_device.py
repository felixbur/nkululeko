"""Unit tests for Model.predict()'s device handling (nkululeko/models/model_tuned.py).

Regression (GH #441, minor): a reloaded model can live on any [FINETUNE]
device (load() now moves it there - see test_model_tuned_load_eval_mode.py),
but predict() built its input tensor with no .to(device) and called
.numpy() directly on the (possibly CUDA) output tensor - silently running
on CPU regardless of the configured device, or crashing with "can't
convert cuda:0 device type tensor to numpy" if it ever did land on GPU.
predict() must place the input on the same device as the model's own
parameters and bring the output back to CPU before converting to numpy.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nkululeko.models.model_tuned import Model


class TestPredictDevice:
    def test_input_moved_to_model_device_and_output_brought_to_cpu(self):
        fake_device = torch.device("cuda:0")

        fake_param = MagicMock()
        fake_param.device = fake_device

        fake_result = MagicMock()
        fake_result.__getitem__.return_value.detach.return_value.cpu.return_value.numpy.return_value = np.array(
            [[0.1, 0.9]]
        )

        moved_tensor = MagicMock()
        fake_input_tensor = MagicMock()
        fake_input_tensor.to.return_value = moved_tensor

        model = MagicMock()
        model.parameters.return_value = iter([fake_param])
        model.side_effect = lambda *_a, **_kw: fake_result

        signal = np.zeros((1, 16000), dtype=np.float32)

        with patch("torch.from_numpy", return_value=fake_input_tensor) as from_numpy:
            result = Model.predict(model, signal)

        from_numpy.assert_called_once_with(signal)
        fake_input_tensor.to.assert_called_once_with(fake_device)
        model.assert_called_once_with(moved_tensor)
        fake_result.__getitem__.return_value.detach.return_value.cpu.assert_called_once()
        np.testing.assert_array_equal(result, np.array([0.1, 0.9]))
