"""RawBoost waveform augmentation.

Tak, Kamble, Patino, Todisco, Evans, "RawBoost: A Raw Data Boosting and
Augmentation Method applied to Automatic Speaker Verification
Anti-Spoofing", ICASSP 2022, pp. 6382-6386.

The noise generation follows the authors' reference implementation
(RawBoost.py in https://github.com/TakHemlata/SSL_Anti-spoofing, MIT
license), because the published results depend on its exact behavior, and
the defaults are those of the arguments of its training script
(main_SSL_LA.py), including `algo = 5`. The differences are robustness
fixes: silence is not divided by its (zero) peak, and the filter band edges
are kept inside (0, sampling rate / 2), so that the default settings also
work for audio that is not sampled at 16 kHz.

Three kinds of noise are available, alone or combined (`algo`):

* 1: linear and non-linear convolutive noise (a random notch filter bank
  applied to the signal and its powers)
* 2: impulsive signal-dependent noise (random samples get a random gain)
* 3: stationary signal-independent noise (colored Gaussian noise at a random
  SNR)

Pure numpy/scipy, no extra dependency. Used by the `rawboost` augmentation
method of `[AUGMENT]` (offline, for any model) and, optionally, on the fly
by the AASIST model. (The reference applies it to the clip it has loaded and
then pads or truncates it; the AASIST model pads or truncates first, so that
the noise is generated for exactly the samples the network sees.)
"""

import copy
from dataclasses import dataclass, fields

import numpy as np
from scipy import signal

# what `algo` stands for
ALGORITHMS = {
    1: "linear and non-linear convolutive noise",
    2: "impulsive signal-dependent noise",
    3: "stationary signal-independent noise",
    4: "1, 2 and 3 in series",
    5: "1 and 2 in series",
    6: "1 and 3 in series",
    7: "2 and 3 in series",
    8: "1 and 2 in parallel",
}

# keep filter band edges this far (in Hz) from 0 and from the Nyquist frequency
_EDGE_HZ = 1 / 1000


@dataclass
class RawBoostParams:
    """Settings of RawBoost. The defaults are those of the reference
    implementation; `algo` selects the combination of noises (see
    `ALGORITHMS`)."""

    algo: int = 5
    n_f: int = 5  # number of harmonics of algorithm 1
    n_bands: int = 5  # number of notch filters per filter bank
    min_f: int = 20  # centre frequency range of the notches (Hz)
    max_f: int = 8000
    min_bw: int = 100  # bandwidth range of the notches (Hz)
    max_bw: int = 1000
    min_coeff: int = 10  # number of taps of a notch filter
    max_coeff: int = 100
    min_g: int = 0  # gain range of a filter bank (dB)
    max_g: int = 0
    min_bias_lin_nonlin: int = 5  # gain bias of the non-linear terms (dB)
    max_bias_lin_nonlin: int = 20
    p: int = 10  # algorithm 2: at most this percent of the samples is changed
    g_sd: int = 2  # algorithm 2: gain of the change
    snr_min: int = 10  # algorithm 3: signal-to-noise ratio range (dB)
    snr_max: int = 40

    def validate(self):
        """Raise a ValueError naming the first setting that makes no sense."""
        if self.algo not in ALGORITHMS:
            raise ValueError(
                f"rawboost_algo = {self.algo} is not one of "
                f"{sorted(ALGORITHMS)} (0, no augmentation, is not an "
                "augmentation method)"
            )
        for name in ("n_f", "n_bands"):
            if getattr(self, name) < 1:
                raise ValueError(f"rawboost_{name} must be >= 1")
        if self.min_coeff < 3:
            raise ValueError("rawboost_min_coeff must be >= 3 (filter taps)")
        if self.min_f < 0 or self.min_bw <= 0:
            raise ValueError("rawboost_min_f must be >= 0 and rawboost_min_bw > 0")
        for low, high in (
            ("min_f", "max_f"),
            ("min_bw", "max_bw"),
            ("min_coeff", "max_coeff"),
            ("min_g", "max_g"),
            ("snr_min", "snr_max"),
        ):
            if getattr(self, low) > getattr(self, high):
                raise ValueError(f"rawboost_{low} is larger than rawboost_{high}")
        if not 0 <= self.p <= 100:
            raise ValueError("rawboost_p is a percentage, 0 to 100")
        return self

    @classmethod
    def from_util(cls, util):
        """Read the `rawboost_*` keys of the [AUGMENT] section."""
        values = {}
        for field in fields(cls):
            key = f"rawboost_{field.name}"
            raw = util.config_val("AUGMENT", key, str(field.default))
            try:
                values[field.name] = int(str(raw).strip())
            except ValueError:
                util.error(f"AUGMENT.{key} = {raw} is not a whole number")
        try:
            return cls(**values).validate()
        except ValueError as e:
            util.error(f"AUGMENT: {e}")


def _rand_range(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    return int(y[0]) if integer else float(y[0])


def _norm_wav(x, always):
    peak = np.amax(abs(x))
    if peak == 0:
        return x
    if always or peak > 1:
        x = x / peak
    return x


def _gen_notch_coeffs(
    n_bands, min_f, max_f, min_bw, max_bw, min_coeff, max_coeff, min_g, max_g, fs
):
    nyquist = fs / 2
    b = 1
    for _ in range(n_bands):
        fc = _rand_range(min_f, max_f, 0)
        bw = _rand_range(min_bw, max_bw, 0)
        c = _rand_range(min_coeff, max_coeff, 1)
        if c % 2 == 0:
            c += 1
        f1 = max(fc - bw / 2, _EDGE_HZ)
        f2 = min(fc + bw / 2, nyquist - _EDGE_HZ)
        if f1 >= f2:
            # the notch lies above the Nyquist frequency (e.g. max_f =
            # 8000 for 8 kHz audio): put it right below it
            f1 = max(_EDGE_HZ, f2 - bw)
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b
        )
    g = _rand_range(min_g, max_g, 0)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, g / 20) * b / np.amax(abs(h))
    return b


def _filter_fir(x, b):
    n = b.shape[0] + 1
    xpad = np.pad(x, (0, n), "constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(n / 2) : int(y.shape[0] - n / 2)]
    return y


def lnl_convolutive_noise(x, params, fs):
    """Linear and non-linear convolutive noise (RawBoost algorithm 1).

    From the second harmonic on the gain range is lowered by the bias (and
    stays lowered for the later ones), as in the reference implementation.
    """
    y = [0] * x.shape[0]
    g_lo, g_hi = params.min_g, params.max_g
    for i in range(params.n_f):
        if i == 1:
            g_lo = params.min_g - params.min_bias_lin_nonlin
            g_hi = params.max_g - params.max_bias_lin_nonlin
        b = _gen_notch_coeffs(
            params.n_bands,
            params.min_f,
            params.max_f,
            params.min_bw,
            params.max_bw,
            params.min_coeff,
            params.max_coeff,
            g_lo,
            g_hi,
            fs,
        )
        y = y + _filter_fir(np.power(x, i + 1), b)
    y = y - np.mean(y)
    return _norm_wav(y, 0)


def isd_additive_noise(x, params):
    """Impulsive signal-dependent noise (RawBoost algorithm 2)."""
    beta = _rand_range(0, params.p, 0)
    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len * (beta / 100))
    idx = np.random.permutation(x_len)[:n]
    f_r = np.multiply(
        (2 * np.random.rand(idx.shape[0])) - 1, (2 * np.random.rand(idx.shape[0])) - 1
    )
    r = params.g_sd * x[idx] * f_r
    y[idx] = x[idx] + r
    return _norm_wav(y, 0)


def ssi_additive_noise(x, params, fs):
    """Stationary signal-independent noise (RawBoost algorithm 3)."""
    noise = np.random.normal(0, 1, x.shape[0])
    b = _gen_notch_coeffs(
        params.n_bands,
        params.min_f,
        params.max_f,
        params.min_bw,
        params.max_bw,
        params.min_coeff,
        params.max_coeff,
        params.min_g,
        params.max_g,
        fs,
    )
    noise = _filter_fir(noise, b)
    noise = _norm_wav(noise, 1)
    snr = _rand_range(params.snr_min, params.snr_max, 0)
    norm = np.linalg.norm(noise, 2)
    if norm == 0:
        return x
    return x + noise / norm * np.linalg.norm(x, 2) / 10.0 ** (0.05 * snr)


def apply_rawboost(x, sr, params):
    """Apply RawBoost to a mono waveform `x` (1-D float array) sampled at
    `sr` Hz. Returns an array of the same length; the result of the additive
    noise (algorithm 3) is not limited to [-1, 1]. The randomness comes from
    numpy's global random state."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"RawBoost needs a mono waveform, got shape {x.shape}")
    algo = params.algo
    if algo == 1:
        return lnl_convolutive_noise(x, params, sr)
    if algo == 2:
        return isd_additive_noise(x, params)
    if algo == 3:
        return ssi_additive_noise(x, params, sr)
    if algo == 4:
        x = lnl_convolutive_noise(x, params, sr)
        x = isd_additive_noise(x, params)
        return ssi_additive_noise(x, params, sr)
    if algo == 5:
        return isd_additive_noise(lnl_convolutive_noise(x, params, sr), params)
    if algo == 6:
        return ssi_additive_noise(lnl_convolutive_noise(x, params, sr), params, sr)
    if algo == 7:
        return ssi_additive_noise(isd_additive_noise(x, params), params, sr)
    if algo == 8:
        x1 = lnl_convolutive_noise(x, params, sr)
        x2 = isd_additive_noise(x, params)
        return _norm_wav(x1 + x2, 0)
    raise ValueError(f"unknown RawBoost algorithm {algo}")
