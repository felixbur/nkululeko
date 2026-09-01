"""Centralized [FINETUNE] config reads for TunedModel (finetuning).

Excludes MODEL.label_smoothing (owned by base Model._get_label_smoothing,
shared with MLP/CNN) and EXP.epochs (shared epoch count, not finetune-specific).
`[MODEL] type = finetune` itself also stays in [MODEL] - it's the model-family
selector shared by every model type, not a finetune-only setting.
"""

import ast
import dataclasses
import typing


def _is_unset(raw):
    """True if a [FINETUNE] value is the missing-key sentinel, or a blank string.

    config_val() returns the raw ini string whenever the key is present,
    even for `key =` (empty) or `key =   ` (whitespace-only) - those are
    non-empty strings in Python and would otherwise reach int()/float() and
    raise ValueError instead of falling back to the intended default.
    """
    return not raw or (isinstance(raw, str) and not raw.strip())


def _parse_balancing(raw):
    """Normalize a [FINETUNE] balancing value.

    config_val() returns the raw ini string when the key is set, so an
    explicit `balancing = False`/`none`/empty string must be treated as
    disabled rather than as a (truthy) non-empty string. config_val_bool()
    doesn't fit here since a real value is an algorithm name (ros/smote/
    adasyn), not a bool.
    """
    if isinstance(raw, str) and raw.strip().lower() in ("false", "none", ""):
        return False
    return raw


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
    num_layers: typing.Optional[int]
    head_layers: typing.Optional[typing.List[int]]
    head_activation: str
    pooling: str
    warmup_ratio: float
    layer_pooling: str

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
        drop = 0.0 if _is_unset(raw_drop) else float(raw_drop)

        push_to_hub = util.config_val_bool("FINETUNE", "push_to_hub", False)
        balancing = _parse_balancing(util.config_val("FINETUNE", "balancing", False))
        pretrained_model = util.config_val(
            "FINETUNE", "pretrained_model", "facebook/wav2vec2-large-robust-ft-swbd-300h"
        )
        class_weight = util.config_val_bool("FINETUNE", "class_weight", False)

        # freeze_layers: number of pretrained encoder layers (counted from
        # the input side) to keep frozen during finetuning. Default 0 keeps
        # today's behavior: the whole backbone trains (only the CNN feature
        # extractor is always frozen, unconditionally, elsewhere).
        raw_freeze_layers = util.config_val("FINETUNE", "freeze_layers", 0)
        freeze_layers = 0 if _is_unset(raw_freeze_layers) else int(raw_freeze_layers)

        # num_layers: total number of transformer encoder layers to build
        # the model with, truncating the pretrained architecture (e.g. use
        # only the first 6 of a 24-layer model). Default None keeps today's
        # behavior: use the pretrained model's full depth.
        raw_num_layers = util.config_val("FINETUNE", "num_layers", False)
        num_layers = None if _is_unset(raw_num_layers) else int(raw_num_layers)

        # head_layers: hidden layer sizes of the classification/regression
        # head sitting on top of the pretrained backbone, matching
        # [MODEL] layers for mlp/mlp_reg so the two approaches' final
        # network can be configured identically for a fair comparison.
        # Default None keeps today's behavior: a single hidden layer sized
        # to the backbone's own hidden_size (config.hidden_size).
        raw_head_layers = util.config_val("FINETUNE", "head_layers", False)
        head_layers = (
            None if _is_unset(raw_head_layers) else ast.literal_eval(raw_head_layers)
        )

        # head_activation: matches [MODEL] activation for mlp/mlp_reg
        # (relu/tanh/sigmoid/leaky_relu). Default "tanh" keeps today's
        # behavior (the head's activation was previously hardcoded to
        # torch.tanh).
        head_activation = util.config_val("FINETUNE", "head_activation", "tanh")

        # pooling: how framewise encoder output is pooled into one vector
        # per utterance before the head. "mean" preserves today's behavior;
        # "meanvar" concatenates mean and variance (doubling the head's
        # input dimension), matching the mean/meanvar pooling ablation
        # nkululeko's own reference finetuning setups grid-search over.
        pooling = util.config_val("FINETUNE", "pooling", "mean")
        if pooling not in ("mean", "meanvar"):
            util.error(f"unknown pooling: {pooling}; expected 'mean' or 'meanvar'")

        # warmup_ratio: fraction of total training steps spent linearly
        # ramping the learning rate up from 0 before the (also linear)
        # decay begins. Default 0.0 keeps today's behavior (HF Trainer's
        # own default - no warmup at all), which lets the full learning
        # rate hit the randomly-initialized head and the pretrained
        # backbone simultaneously from step 1.
        warmup_ratio = float(util.config_val("FINETUNE", "warmup_ratio", "0.0"))

        # layer_pooling: which encoder layer(s) feed the (time-)pooling step
        # above. "last" preserves today's behavior (only the final encoder
        # layer's hidden states). "weighted" adds one learnable scalar per
        # encoder layer (softmax-normalized across layers) and pools a
        # weighted sum of every layer's hidden states instead - the
        # SUPERB-benchmark "weighted sum of hidden states" technique, since
        # useful paralinguistic/prosodic signal is often concentrated in
        # earlier or middle layers rather than the last one.
        layer_pooling = util.config_val("FINETUNE", "layer_pooling", "last")
        if layer_pooling not in ("last", "weighted"):
            util.error(
                f"unknown layer_pooling: {layer_pooling}; expected 'last' or 'weighted'"
            )

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
            class_weight=class_weight,
            measure=measure,
            pretrained_model=pretrained_model,
            freeze_layers=freeze_layers,
            num_layers=num_layers,
            head_layers=head_layers,
            head_activation=head_activation,
            pooling=pooling,
            warmup_ratio=warmup_ratio,
            layer_pooling=layer_pooling,
        )
