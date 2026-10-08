"""Centralized [AASIST] config reads for AasistModel.

Mirrors nkululeko/models/finetune_config.py's pattern: one dataclass, one
from_util() classmethod, so every [AASIST] key is resolved in one place.
MODEL.learning_rate / MODEL.optimizer / MODEL.weight_decay / MODEL.loss /
MODEL.class_weight / MODEL.patience / EXP.epochs stay in their existing
shared sections rather than [AASIST] -- they're read the same way ADM
reads them, for direct comparability between the two model types.
"""

import dataclasses

from nkululeko.models.domain_adversarial import DannConfig


@dataclasses.dataclass
class AasistConfig:
    """Resolved [AASIST] settings, one field per config key."""

    device: str
    ssl_model: str
    max_len: int
    batch_size: int
    ssl_layer_pooling: str
    freeze_ssl_frontend: bool
    dann: DannConfig

    @classmethod
    def from_util(cls, util) -> "AasistConfig":
        """Build from an experiment Util, resolving all [AASIST] keys.

        `util` is a plain parameter (not `self.util`), matching
        FinetuneConfig.from_util, so this stays a pure function - testable
        without an AasistModel/experiment.

        device reads from the shared [MODEL] section (not [AASIST]), like
        every other shared MODEL.* key this class deliberately does not
        duplicate (see module docstring); AasistModel takes self.device
        from self.cfg.device.
        """
        import torch

        raw_device = util.config_val("MODEL", "device", False)
        device = (
            raw_device
            if raw_device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ssl_model: the HuggingFace frontend checkpoint. xls-r-300m matches
        # the upstream AASIST paper's frontend (via fairseq there); swapped
        # to HuggingFace's Wav2Vec2Model here to avoid a fairseq dependency
        # (see model_aasist_core.py's HFWav2Vec2Frontend docstring).
        ssl_model = util.config_val(
            "AASIST", "ssl_model", "facebook/wav2vec2-xls-r-300m"
        )

        # max_len: fixed waveform length in samples every clip is
        # padded/truncated to. 64600 (~4.0375s at 16kHz) matches upstream's
        # own default, tuned for ASVspoof-style short utterances.
        max_len = int(util.config_val("AASIST", "max_len", "64600"))

        batch_size = int(util.config_val("AASIST", "batch_size", "24"))

        # ssl_layer_pooling: "last" (default, only the final encoder layer)
        # or "weighted" (a learnable softmax-normalized combination of every hidden
        # state layer -- see HFWav2Vec2Frontend's docstring).
        ssl_layer_pooling = util.config_val("AASIST", "ssl_layer_pooling", "last")
        if ssl_layer_pooling not in ("last", "weighted"):
            util.error(
                f"unknown AASIST.ssl_layer_pooling: {ssl_layer_pooling}; "
                "expected 'last' or 'weighted'"
            )

        # freeze_ssl_frontend: skip training the SSL frontend's own
        # parameters entirely (see HFWav2Vec2Frontend's docstring for why
        # this is also a speed win, not just a regularization choice).
        freeze_ssl_frontend = util.config_val_bool(
            "AASIST", "freeze_ssl_frontend", False
        )

        # Domain-adversarial training keys, shared with other models.
        dann = DannConfig.from_util(util)

        return cls(
            device=device,
            ssl_model=ssl_model,
            max_len=max_len,
            batch_size=batch_size,
            ssl_layer_pooling=ssl_layer_pooling,
            freeze_ssl_frontend=freeze_ssl_frontend,
            dann=dann,
        )
