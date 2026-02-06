import numpy as np
import pytest

from nkululeko.autopredict.estimate_snr import SNREstimator


class TestSNREstimator:
    def test_init(self):
        signal = np.random.randn(16000)
        estimator = SNREstimator(signal, 16000, window_size=320, hop_size=160)
        
        assert estimator.sample_rate == 16000
        assert estimator.frame_length == 320
        assert estimator.hop_length == 160
        assert len(estimator.audio_data) == 16000

    def test_frame_audio(self):
        signal = np.arange(1000)
        estimator = SNREstimator(signal, 16000, window_size=100, hop_size=50)
        
        frames = estimator.frame_audio(signal)
        
        # Calculate expected number of frames
        expected_num_frames = 1 + (len(signal) - 100) // 50
        assert len(frames) == expected_num_frames
        assert len(frames[0]) == 100

    def test_calculate_log_energy(self):
        signal = np.random.randn(16000)
        estimator = SNREstimator(signal, 16000)
        
        frame = np.array([0.1, 0.2, 0.3, 0.4])
        log_energy = estimator.calculate_log_energy(frame)
        
        expected_energy = np.sum(frame**2)
        expected_log_energy = np.log(expected_energy)
        assert log_energy == pytest.approx(expected_log_energy)

    def test_calculate_snr(self):
        signal = np.random.randn(16000)
        estimator = SNREstimator(signal, 16000)
        
        energy_high = 100
        energy_low = 10
        snr = estimator.calculate_snr(energy_high, energy_low)
        
        expected_snr = 10 * np.log10(energy_high / energy_low)
        assert snr == pytest.approx(expected_snr)

    def test_estimate_snr(self):
        # Create a signal with clear high and low energy regions
        signal = np.concatenate([
            np.random.randn(8000) * 0.1,  # Low energy
            np.random.randn(8000) * 1.0   # High energy
        ])
        
        estimator = SNREstimator(signal, 16000, window_size=320, hop_size=160)
        
        estimated_snr, log_energies, threshold_low, threshold_high = estimator.estimate_snr()
        
        assert isinstance(estimated_snr, (float, np.floating))
        assert len(log_energies) > 0
        assert threshold_low < threshold_high
        # SNR should be positive for this signal
        assert estimated_snr > 0

    def test_estimate_snr_silent_signal(self):
        # Test with very low energy signal
        signal = np.random.randn(16000) * 0.001
        estimator = SNREstimator(signal, 16000)
        
        estimated_snr, log_energies, threshold_low, threshold_high = estimator.estimate_snr()
        
        assert isinstance(estimated_snr, (float, np.floating))
        assert not np.isnan(estimated_snr)
        assert not np.isinf(estimated_snr)
