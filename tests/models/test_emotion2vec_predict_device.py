"""Unit tests for Emotion2vecModel.predict()'s device handling
(nkululeko/models/model_tuned.py). See test_model_predict_device.py for the
identical fix already applied to the sibling Model.predict() (GH #441).

Regression (GH #445): before PR #442, TunedModel.load() never moved a
reloaded model off CPU regardless of [FINETUNE] device, so this mismatch
never actually triggered. #442 fixed load() to call self.model.to(self.
device) for the emotion2vec_backbone branch too, so a reloaded
Emotion2vecModel's weights now genuinely live on GPU with device = cuda
configured - but predict() built its input tensor with no .to(device),
which raises "Expected all tensors to be on the same device" as soon as
that actually happens. predict() must place the input on the same device
as the model's own parameters.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from nkululeko.models.model_tuned import Emotion2vecModel


class TestEmotion2vecPredictDevice:
    def test_ndarray_input_is_moved_to_model_device(self):
        fake_device = torch.device("cuda:0")

        fake_param = MagicMock()
        fake_param.device = fake_device

        fake_logits = MagicMock()
        fake_logits.detach.return_value.cpu.return_value.numpy.return_value = np.array(
            [[0.1, 0.9]]
        )
        fake_result = MagicMock(logits=fake_logits)

        unsqueezed_tensor = MagicMock()
        moved_tensor = MagicMock()
        unsqueezed_tensor.to.return_value = moved_tensor
        fake_input_tensor = MagicMock()
        fake_input_tensor.unsqueeze.return_value = unsqueezed_tensor

        model = MagicMock()
        model.is_classifier = True
        model.parameters.return_value = iter([fake_param])
        model.side_effect = lambda *_a, **_kw: fake_result

        signal = np.zeros((16000,), dtype=np.float32)

        with patch("torch.from_numpy", return_value=fake_input_tensor) as from_numpy:
            result = Emotion2vecModel.predict(model, signal)

        from_numpy.assert_called_once_with(signal)
        fake_input_tensor.unsqueeze.assert_called_once_with(0)
        unsqueezed_tensor.to.assert_called_once_with(fake_device)
        model.assert_called_once_with(moved_tensor)
        fake_logits.detach.return_value.cpu.assert_called_once()
        np.testing.assert_array_equal(result, np.array([0.1, 0.9]))

    def test_tensor_input_is_moved_to_model_device(self):
        """The non-ndarray branch (already a tensor) must also follow the
        model's device, not just the torch.from_numpy path."""
        fake_device = torch.device("cuda:0")

        fake_param = MagicMock()
        fake_param.device = fake_device

        fake_logits = MagicMock()
        fake_logits.detach.return_value.cpu.return_value.numpy.return_value = np.array(
            [[0.2, 0.8]]
        )
        fake_result = MagicMock(logits=fake_logits)

        moved_tensor = MagicMock()
        signal_tensor = MagicMock()
        signal_tensor.dim.return_value = 1
        signal_tensor.unsqueeze.return_value.to.return_value = moved_tensor

        model = MagicMock()
        model.is_classifier = False
        model.parameters.return_value = iter([fake_param])
        model.side_effect = lambda *_a, **_kw: fake_result

        result = Emotion2vecModel.predict(model, signal_tensor)

        signal_tensor.unsqueeze.assert_called_once_with(0)
        signal_tensor.unsqueeze.return_value.to.assert_called_once_with(fake_device)
        model.assert_called_once_with(moved_tensor)
        np.testing.assert_array_equal(result, np.array([0.2, 0.8]))
