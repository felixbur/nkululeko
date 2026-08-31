"""Unit tests for TunedModel._normalize_signal (nkululeko/models/model_tuned.py).

Regression: data_collator() (used for every training and HF-internal-eval
batch) always runs raw audio through self.processor first -
Wav2Vec2FeatureExtractor's do_normalize=True zero-mean/unit-variance
normalization. get_predictions() (the dev/test report path) and
predict_sample() (the public single-sample/demo path) instead fed
audiofile.read()'s raw signal straight into Model.predict(), skipping that
normalization entirely - the model was finetuned to expect normalized input,
so every "final" dev/test score was computed against wrong-scale audio.
Confirmed empirically: fixing the separate missing-eval()-mode bug alone
left dev CCC far below the live training-time eval score; this was the
second, larger contributor to that gap.
"""

import types

import numpy as np

from nkululeko.models.model_tuned import TunedModel


class FakeProcessor:
    """Stands in for Wav2Vec2Processor: do_normalize-style zero-mean/unit-variance."""

    def __call__(self, signal, sampling_rate, padding):
        signal = np.asarray(signal, dtype=np.float32)
        normalized = (signal - signal.mean()) / (signal.std() + 1e-7)
        return {"input_values": [normalized]}


def make_fake_self(processor):
    return types.SimpleNamespace(processor=processor, sampling_rate=16000)


class TestNormalizeSignal:
    def test_none_processor_returns_signal_unchanged(self):
        fake_self = make_fake_self(processor=None)
        signal = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = TunedModel._normalize_signal(fake_self, signal)
        assert result is signal

    def test_applies_processor_normalization(self):
        fake_self = make_fake_self(processor=FakeProcessor())
        # A signal with a large, nonzero mean and scale - if normalization
        # is skipped, the mean/std below would stay ~50/10 instead of ~0/1.
        signal = np.random.RandomState(0).normal(loc=50.0, scale=10.0, size=1000)

        result = TunedModel._normalize_signal(fake_self, signal)

        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.1

    def test_squeezes_extra_dimensions_before_normalizing(self):
        # audiofile.read(..., always_2d=True) returns shape (channels,
        # samples) even for mono audio - must be squeezed the same way
        # data_collator() squeezes before handing audio to the processor.
        fake_self = make_fake_self(processor=FakeProcessor())
        signal = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)  # (1, 4)

        result = TunedModel._normalize_signal(fake_self, signal)

        assert result.ndim == 1
        assert result.shape == (4,)
