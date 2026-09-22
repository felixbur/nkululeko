"""VoiceSauce-style voice-quality features (issue #422).

VoiceSauce (Shue et al., 2011, MATLAB) and its Python port opensauce-python
(https://github.com/voicesauce/opensauce-python) are what phoneticians
usually mean by "voice sauce". opensauce-python turned out to be abandoned
(last commit 2019) and, despite the README, only implements F0/formants/SHR
-- not the harmonic-amplitude voice-quality measures (H1, H2, H1-H2, ...)
and CPP that VoiceSauce is actually cited for; those are still listed as
unimplemented in its own TODO.md. So instead of wrapping that CLI tool, this
module computes VoiceSauce's signature measures directly via parselmouth
(Praat), the same dependency feats_praat_core.py already uses:

* H1, H2, H4: amplitude (dB) of the first, second, and fourth harmonic of F0.
* A1, A2, A3: amplitude (dB) near the first three formants.
* H1H2, H1A1, H1A2, H1A3, H2H4: the corresponding spectral-tilt/voice-quality
  differences (breathy voices tend to have a higher H1-H2 and H1-A3; creaky
  voices a lower one).
* CPP: cepstral peak prominence (smoothed), via Praat's built-in
  PowerCepstrogram -- a measure of periodicity strength, low in
  breathy/creaky voice.

H1/H2/H4/A1/A2/A3 are estimated per glottal-pulse-aligned analysis point
(reusing the same point_process points feats_praat_core.py uses for
jitter/shimmer) as the local spectral energy in a narrow band around the
target harmonic/formant frequency, then averaged across the file -- not the
LPC-inverse-filtered, formant-bandwidth-corrected estimate VoiceSauce itself
uses, which needs the phoneme-level formant-tracking accuracy VoiceSauce's
MATLAB pipeline assumes. This is a documented simplification, not a bug: the
band-energy approach is robust to the same F0/formant tracking imprecision
that already affects every other measure in this file, at the cost of not
correcting for formant bandwidth.

SHR (subharmonic-to-harmonic ratio) is intentionally not included here.
opensauce-python's own SHR implementation is pure Python/NumPy (no MATLAB or
external binary needed) and could in principle be vendored, but its authors'
own comments flag known quirks in the algorithm (e.g. an acknowledged
off-by-one "bug ... presumably in the matlab code" they chose to emulate
rather than fix), and it needs a distinct millisecond-based frame_length/
timestep/datalen calling convention that would need its own validation
against reference values this project doesn't have. Left as a follow-up
rather than integrated without being able to verify it's correct.
"""

import math

import audiofile
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

# Half-width, in Hz, of the band searched around each target harmonic/formant
# frequency for its spectral energy -- wide enough to tolerate F0/formant
# tracking imprecision, narrow enough not to bleed into a neighboring
# harmonic at typical human F0 (>= ~50 Hz).
BAND_HALF_WIDTH_HZ = 25
# Width, in seconds, of the Gaussian-windowed segment analyzed at each point.
ANALYSIS_WINDOW_S = 0.025


class VoiceSauceFeatureExtractor:
    """Extract VoiceSauce-style voice-quality features from a single sound."""

    def __init__(self, f0min=75, f0max=300):
        self.f0min = f0min
        self.f0max = f0max

    def extract_all_features(self, sound):
        """Extract all VoiceSauce-style features from a single sound object."""
        pitch = call(sound, "To Pitch", 0.0, self.f0min, self.f0max)
        formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
        point_process = call(
            sound, "To PointProcess (periodic, cc)", self.f0min, self.f0max
        )

        harmonic_features = self._extract_harmonic_amplitude_features(
            sound, pitch, formants, point_process
        )
        cpp_feature = self._extract_cpp_feature(sound)

        return {**harmonic_features, **cpp_feature}

    def _band_amplitude_db(self, spectrum, center_freq):
        """Local spectral energy (dB) in a narrow band around center_freq."""
        low = max(center_freq - BAND_HALF_WIDTH_HZ, 0)
        high = center_freq + BAND_HALF_WIDTH_HZ
        try:
            energy = call(spectrum, "Get band energy", low, high)
        except parselmouth.PraatError:
            return np.nan
        if energy is None or math.isnan(energy) or energy <= 0:
            return np.nan
        return 10 * math.log10(energy)

    def _extract_harmonic_amplitude_features(
        self, sound, pitch, formants, point_process
    ):
        """H1/H2/H4/A1/A2/A3 and their VoiceSauce spectral-tilt differences."""
        num_points = call(point_process, "Get number of points")

        h1_vals, h2_vals, h4_vals = [], [], []
        a1_vals, a2_vals, a3_vals = [], [], []
        h1h2_vals, h1a1_vals, h1a2_vals, h1a3_vals, h2h4_vals = [], [], [], [], []

        duration = sound.get_total_duration()
        for point in range(num_points):
            t = call(point_process, "Get time from index", point + 1)
            f0 = call(pitch, "Get value at time", t, "Hertz", "Linear")
            if math.isnan(f0) or f0 <= 0:
                continue

            half_window = ANALYSIS_WINDOW_S / 2
            start = max(t - half_window, 0)
            end = min(t + half_window, duration)
            if end <= start:
                continue
            try:
                part = sound.extract_part(
                    start, end, parselmouth.WindowShape.GAUSSIAN1, 2.0, False
                )
                spectrum = part.to_spectrum()
            except parselmouth.PraatError:
                continue

            f1 = call(formants, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formants, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formants, "Get value at time", 3, t, "Hertz", "Linear")

            h1 = self._band_amplitude_db(spectrum, f0)
            h2 = self._band_amplitude_db(spectrum, 2 * f0)
            h4 = self._band_amplitude_db(spectrum, 4 * f0)
            a1 = self._band_amplitude_db(spectrum, f1) if not math.isnan(f1) else np.nan
            a2 = self._band_amplitude_db(spectrum, f2) if not math.isnan(f2) else np.nan
            a3 = self._band_amplitude_db(spectrum, f3) if not math.isnan(f3) else np.nan

            if not math.isnan(h1):
                h1_vals.append(h1)
            if not math.isnan(h2):
                h2_vals.append(h2)
            if not math.isnan(h4):
                h4_vals.append(h4)
            if not math.isnan(a1):
                a1_vals.append(a1)
            if not math.isnan(a2):
                a2_vals.append(a2)
            if not math.isnan(a3):
                a3_vals.append(a3)

            # Per-point differences (VoiceSauce's own convention), not
            # differences of the file-level means computed below.
            if not math.isnan(h1) and not math.isnan(h2):
                h1h2_vals.append(h1 - h2)
            if not math.isnan(h1) and not math.isnan(a1):
                h1a1_vals.append(h1 - a1)
            if not math.isnan(h1) and not math.isnan(a2):
                h1a2_vals.append(h1 - a2)
            if not math.isnan(h1) and not math.isnan(a3):
                h1a3_vals.append(h1 - a3)
            if not math.isnan(h2) and not math.isnan(h4):
                h2h4_vals.append(h2 - h4)

        def _mean(values):
            return float(np.mean(values)) if values else np.nan

        return {
            "H1": _mean(h1_vals),
            "H2": _mean(h2_vals),
            "H4": _mean(h4_vals),
            "A1": _mean(a1_vals),
            "A2": _mean(a2_vals),
            "A3": _mean(a3_vals),
            "H1H2": _mean(h1h2_vals),
            "H1A1": _mean(h1a1_vals),
            "H1A2": _mean(h1a2_vals),
            "H1A3": _mean(h1a3_vals),
            "H2H4": _mean(h2h4_vals),
        }

    def _extract_cpp_feature(self, sound):
        """Cepstral peak prominence, smoothed (CPPS), via Praat's built-in
        PowerCepstrogram -- low in breathy/creaky voice, high in modal voice.
        """
        try:
            power_cepstrogram = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
            cpp = call(
                power_cepstrogram,
                "Get CPPS",
                "no",
                0.01,
                0.001,
                60,
                330,
                0.05,
                "Parabolic",
                0.001,
                0,
                "Straight",
                "Robust",
            )
        except parselmouth.PraatError:
            cpp = np.nan
        if cpp is None or (isinstance(cpp, float) and math.isnan(cpp)):
            cpp = np.nan
        return {"CPP": cpp}


_FEATURE_NAMES = [
    "H1",
    "H2",
    "H4",
    "A1",
    "A2",
    "A3",
    "H1H2",
    "H1A1",
    "H1A2",
    "H1A3",
    "H2H4",
    "CPP",
]


def compute_features(file_index):
    """Compute VoiceSauce-style features for every (file, start, end) in
    file_index, returning a dataframe with one row per entry."""
    extractor = VoiceSauceFeatureExtractor()
    feature_list = []

    for wave_file, start, end in tqdm(file_index.to_list()):
        try:
            signal, sampling_rate = audiofile.read(
                wave_file,
                offset=start.total_seconds(),
                duration=(end - start).total_seconds(),
                always_2d=True,
            )
            sound = parselmouth.Sound(values=signal, sampling_frequency=sampling_rate)
            features = extractor.extract_all_features(sound)
        except Exception as error:
            print(f"error on file {wave_file}: {error}")
            features = {key: np.nan for key in _FEATURE_NAMES}
        feature_list.append(features)

    return pd.DataFrame(feature_list)
