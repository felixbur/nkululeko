"""Centralized [RNN] config reads, shared by every RNN-cell-based model type.

Mirrors nkululeko/models/aasist_config.py's pattern: one dataclass, one
from_util() classmethod, so every [RNN] key is resolved in one place.
MODEL.learning_rate / MODEL.optimizer / MODEL.weight_decay / MODEL.loss /
MODEL.class_weight / MODEL.patience / MODEL.drop stay in their existing
shared [MODEL] section rather than [RNN] -- they're read the same way
every other ANN model reads them, for direct comparability between model
types. The input frames come from the feature pipeline (FEATS.type with
FEATS.level = frame), not from [RNN].

Named generically ([RNN], not [LSTM]): every key here (hidden_size,
num_layers, bidirectional, batch_size) is equally meaningful for any RNN
cell type. MODEL.type itself stays "lstm"/"lstm_reg" for the model types
this actually builds; a future "gru" would read the very same [RNN]
section.
"""

import dataclasses

from nkululeko.models.domain_adversarial import DannConfig


@dataclasses.dataclass
class RNNConfig:
    """Resolved [RNN] settings, one field per config key."""

    device: str
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
        without a model/experiment.

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
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_size=batch_size,
            dann=dann,
        )
