"""Centralized [RNN] config reads, shared by every RNN-cell-based model type.

Mirrors nkululeko/models/aasist_config.py's pattern: one dataclass, one
from_util() classmethod, so every [RNN] key is resolved in one place.
MODEL.learning_rate / MODEL.optimizer / MODEL.weight_decay / MODEL.loss /
MODEL.class_weight / MODEL.patience / MODEL.drop stay in their existing
shared [MODEL] section rather than [RNN] -- they're read the same way
every other ANN model reads them, for direct comparability between model
types.

Named generically ([RNN], not [LSTM]): every key here (the frame source,
its parameters, hidden_size/num_layers/bidirectional, batch_size) is
equally meaningful for any RNN cell type, not specific to LSTM. MODEL.type
itself stays "lstm"/"lstm_reg" for the model types this actually builds;
a future "gru" (or an explicit "blstm", though bidirectional=True on
MODEL.type=lstm already covers that case) would read the very same [RNN]
section rather than duplicating it.
"""

import dataclasses

from nkululeko.models.domain_adversarial import DannConfig

_TARGET_SR = 16000

AVAILABLE_FEATURE_SOURCES = ("melspec", "opensmile_lld")

# Same allow-list feats_opensmile.py's FEATS.set validates against.
AVAILABLE_OPENSMILE_SETS = ("eGeMAPSv02", "ComParE_2016", "GeMAPSv01a", "eGeMAPSv01a")


@dataclasses.dataclass
class RNNConfig:
    """Resolved [RNN] settings, one field per config key."""

    device: str
    feature_source: str
    n_mels: int
    win_samples: int
    hop_samples: int
    opensmile_set: str
    max_len: int
    hidden_size: int
    num_layers: int
    bidirectional: bool
    dropout: float
    batch_size: int
    dann: DannConfig

    @classmethod
    def from_util(cls, util) -> "RNNConfig":
        """Build from an experiment Util, resolving all [RNN] keys.

        `util` is a plain parameter (not `self.util`), matching
        AasistConfig.from_util, so this stays a pure function - testable
        without an LSTMModel/experiment.

        device reads from the shared [MODEL] section (not [RNN]), like
        every other shared MODEL.* key this class deliberately does not
        duplicate (see module docstring).
        """
        import torch

        raw_device = util.config_val("MODEL", "device", False)
        device = (
            raw_device
            if raw_device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        feature_source = util.config_val("RNN", "feature_source", "melspec")
        if feature_source not in AVAILABLE_FEATURE_SOURCES:
            util.error(
                f"unknown RNN.feature_source: {feature_source}; "
                f"expected one of {AVAILABLE_FEATURE_SOURCES}"
            )

        # n_mels/win_dur/hop_dur only matter for feature_source=melspec, but
        # are always resolved (cheap, keeps from_util a single straight-line
        # read instead of branching on feature_source twice).
        n_mels = int(util.config_val("RNN", "n_mels", "64"))
        win_dur = float(util.config_val("RNN", "win_dur", "25"))  # ms
        hop_dur = float(util.config_val("RNN", "hop_dur", "10"))  # ms
        win_samples = round(win_dur * _TARGET_SR / 1000)
        hop_samples = round(hop_dur * _TARGET_SR / 1000)

        opensmile_set = util.config_val("RNN", "opensmile_set", "eGeMAPSv02")
        if opensmile_set not in AVAILABLE_OPENSMILE_SETS:
            util.warn(
                f"RNN.opensmile_set '{opensmile_set}' might not be supported. "
                f"Available sets: {', '.join(AVAILABLE_OPENSMILE_SETS)}"
            )

        # max_len: fixed waveform length in samples every clip is
        # padded/truncated to before framing -- keeps every sample's frame
        # count T identical regardless of feature_source, so the default
        # DataLoader collate works unmodified (no custom collate_fn/
        # pack_padded_sequence). 64600 (~4.0375s @16kHz) matches
        # [AASIST] max_len's default, the one other raw-audio model type.
        max_len = int(util.config_val("RNN", "max_len", "64600"))

        hidden_size = int(util.config_val("RNN", "hidden_size", "64"))
        num_layers = int(util.config_val("RNN", "num_layers", "2"))
        bidirectional = util.config_val_bool("RNN", "bidirectional", False)

        # dropout reuses the shared MODEL.drop key (same key every layered
        # ANN model reads), not a separate [RNN] key.
        drop = util.config_val("MODEL", "drop", "False")
        dropout = 0.0 if str(drop).strip().lower() == "false" else float(drop)

        batch_size = int(util.config_val("RNN", "batch_size", "32"))

        # Domain-adversarial training keys, shared with other models.
        dann = DannConfig.from_util(util)

        return cls(
            device=device,
            feature_source=feature_source,
            n_mels=n_mels,
            win_samples=win_samples,
            hop_samples=hop_samples,
            opensmile_set=opensmile_set,
            max_len=max_len,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_size=batch_size,
            dann=dann,
        )
