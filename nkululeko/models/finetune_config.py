"""Centralized [FINETUNE] config reads for TunedModel (finetuning).

Excludes MODEL.label_smoothing (owned by base Model._get_label_smoothing,
shared with MLP/CNN) and EXP.epochs (shared epoch count, not finetune-specific).
`[MODEL] type = finetune` itself also stays in [MODEL] - it's the model-family
selector shared by every model type, not a finetune-only setting.
"""

import dataclasses
import typing


@dataclasses.dataclass
class FinetuneConfig:
    """Resolved [FINETUNE] settings, one field per config key."""

    device: str
    batch_size: int
    learning_rate: float
    max_duration: float
    drop: float
    push_to_hub: bool
    balancing: typing.Union[str, bool]
    loss: str
    class_weight: bool
    measure: str
    pretrained_model: str
    freeze_layers: int

    @classmethod
    def from_util(cls, util, is_classifier: bool) -> "FinetuneConfig":
        """Build from an experiment Util, resolving all [FINETUNE] keys.

        `util` is a plain parameter (not `self.util`) so this stays a pure
        function of (util, is_classifier) - testable without a TunedModel.
        Each config_val*() call below is 3-positional/no-kwargs so
        scripts/gen_defaults_table.py's AST scan keeps picking it up.
        """
        raw_device = util.config_val("FINETUNE", "device", False)
        if not raw_device:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = raw_device

        batch_size = int(util.config_val("FINETUNE", "batch_size", "8"))
        learning_rate = float(util.config_val("FINETUNE", "learning_rate", "0.0001"))
        max_duration = float(util.config_val("FINETUNE", "max_duration", "8.0"))

        raw_drop = util.config_val("FINETUNE", "drop", False)
        drop = float(raw_drop) if raw_drop else 0.1

        push_to_hub = util.config_val_bool("FINETUNE", "push_to_hub", False)
        balancing = util.config_val("FINETUNE", "balancing", False)
        pretrained_model = util.config_val(
            "FINETUNE", "pretrained_model", "facebook/wav2vec2-large-robust-ft-swbd-300h"
        )
        class_weight = util.config_val("FINETUNE", "class_weight", False)

        # freeze_layers: number of pretrained encoder layers (counted from
        # the input side) to keep frozen during finetuning. Default 0 keeps
        # today's behavior: the whole backbone trains (only the CNN feature
        # extractor is always frozen, unconditionally, elsewhere).
        freeze_layers = int(util.config_val("FINETUNE", "freeze_layers", 0))

        # loss & measure: default depends on task type. measure is NOT
        # configurable for classification (unchanged from prior behavior).
        if is_classifier:
            loss = util.config_val("FINETUNE", "loss", "cross")
            measure = "uar"
        else:
            loss = util.config_val("FINETUNE", "loss", "1-ccc")
            measure = util.config_val("FINETUNE", "measure", "ccc")

        return cls(
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_duration=max_duration,
            drop=drop,
            push_to_hub=bool(push_to_hub),
            balancing=balancing,
            loss=loss,
            class_weight=bool(class_weight),
            measure=measure,
            pretrained_model=pretrained_model,
            freeze_layers=freeze_layers,
        )
