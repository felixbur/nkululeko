# mlp_layers.py
"""Hidden-layer stack shared by `mlp`/`mlp_reg` (model_mlp.py /
model_mlp_regression.py), which used to each carry their own copy."""

import ast
from collections import OrderedDict

import torch


def _check_drop_value(value, util):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        util.error(f"invalid value for dropout: {value!r} (must be a number)")
    if not (0 <= value < 1):
        util.error(f"invalid value for dropout: {value} (must be in [0, 1))")


def parse_drop_config(util, num_layers=None):
    """Parse `[MODEL] drop` into `False`, a float in [0, 1), or a list of
    per-layer floats in [0, 1).

    `config_val` returns the raw ini string, so a plain `float(drop)`
    crashes on `"False"` or `"[0.3, 0.2]"`. `num_layers`, if given, warns
    when a per-layer list doesn't match the actual hidden-layer count.
    """
    drop = util.config_val("MODEL", "drop", "False")
    if drop == "False":
        return False
    try:
        drop = ast.literal_eval(drop)
    except (ValueError, SyntaxError):
        # if it's not a list, it should be a float
        try:
            drop = float(drop)
        except ValueError:
            util.error(f"invalid value for dropout: {drop}")

    if isinstance(drop, bool):
        # True/False as a literal (not the "False" string above, already
        # handled) is ambiguous: reject rather than silently becoming 1.0
        util.error(f"invalid value for dropout: {drop} (use a number, not True/False)")
    if isinstance(drop, tuple):
        drop = list(drop)
    if isinstance(drop, list):
        for value in drop:
            _check_drop_value(value, util)
        if num_layers is not None and len(drop) != num_layers:
            util.warn(
                f"[MODEL] drop has {len(drop)} value(s) for {num_layers} "
                "hidden layer(s): extra values are ignored, layers beyond "
                "the list get no dropout"
            )
    else:
        # reject before float(): a quoted string ("'0.3'") would otherwise
        # be silently accepted, and None/a dict/a set would raise a raw
        # TypeError/ValueError instead of util.error()
        _check_drop_value(drop, util)
        drop = float(drop)
    return drop


def load_mlp_state_dict(model, path, util, map_location=None):
    """load_state_dict(), turning a missing/unreadable file or a
    hidden-layer key mismatch into a clear util.error() instead of a raw
    traceback.

    `map_location` should be the model's own device, so loading a
    CUDA-trained checkpoint on a CPU-only machine doesn't itself raise
    (that RuntimeError isn't a key mismatch, so it gets no 1.14.3 hint). A
    "Missing key(s)"/"Unexpected key(s)" RuntimeError most likely means
    either `path` was saved before nkululeko 1.14.3, whose mlp/mlp_reg
    hidden-layer layout changed (see CHANGELOG), or `[MODEL] layers` no
    longer matches what the checkpoint was trained with.
    """
    try:
        state_dict = torch.load(path, map_location=map_location)
    except FileNotFoundError:
        util.error(f"model file not found: {path}")
    except Exception as e:
        # e.g. a truncated/empty file (interrupted training, disk full)
        # raises EOFError or pickle.UnpicklingError, not FileNotFoundError
        util.error(f"could not read model file {path}: {e}")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as re:
        hint = ""
        if "Missing key(s)" in str(re) or "Unexpected key(s)" in str(re):
            hint = (
                "\nThis may be a checkpoint saved before nkululeko 1.14.3, "
                "whose mlp/mlp_reg layer layout changed, or [MODEL] layers "
                "may differ from the one it was trained with; retrain the "
                "model or fix the config to match."
            )
        util.error(f"could not load model weights from {path}: {re}{hint}")


class MLPBlock(torch.nn.Module):
    """i -> layers[0] -> ... -> layers[-1] -> o, with `activation` after
    each hidden Linear and, if `drop`, a Dropout after that.

    `layers` (a list or a dict, from `[MODEL] layers`) is used in the
    order given, not resorted by width. `drop` is `False`, a single
    number (every hidden layer), or a list (positional, one per layer).
    """

    def __init__(self, i, layers, o, drop, activation):
        super().__init__()
        layer_widths = layers if isinstance(layers, list) else list(layers.values())
        self.feat_dim = layer_widths[-1]
        net = OrderedDict()
        in_dim = i
        for idx, width in enumerate(layer_widths):
            net[str(idx)] = torch.nn.Linear(in_dim, width)
            net[f"{idx}_r"] = activation
            if drop:
                if isinstance(drop, list):
                    if idx < len(drop):
                        net[f"{idx}_d"] = torch.nn.Dropout(float(drop[idx]))
                else:
                    net[f"{idx}_d"] = torch.nn.Dropout(float(drop))
            in_dim = width
        net[str(len(layer_widths))] = torch.nn.Linear(in_dim, o)
        self.linear = torch.nn.Sequential(net)

    def forward(self, x, return_features=False):
        # x: (batch_size, channels, samples)
        x = x.squeeze(dim=1).float()
        if not return_features:
            return self.linear(x)
        # input to the output layer: where DANN heads attach
        hidden = self.linear[:-1](x)
        return self.linear[-1](hidden), hidden
