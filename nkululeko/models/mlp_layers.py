# mlp_layers.py
"""Hidden-layer stack shared by `mlp`/`mlp_reg` (model_mlp.py /
model_mlp_regression.py), which used to each carry their own copy."""

import ast
from collections import OrderedDict

import torch


def _check_drop_value(value, util):
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
        drop = float(drop)
        _check_drop_value(drop, util)
    return drop


def load_mlp_state_dict(model, path, util):
    """load_state_dict(), turning a missing file or a key mismatch into a
    clear util.error() instead of a raw traceback.

    A key mismatch most likely means `path` was saved before nkululeko
    1.14.3, whose mlp/mlp_reg hidden-layer layout changed (see CHANGELOG).
    """
    try:
        state_dict = torch.load(path)
    except FileNotFoundError:
        util.error(f"model file not found: {path}")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as re:
        util.error(
            f"could not load model weights from {path}: {re}\n"
            "This may be a checkpoint saved before nkululeko 1.14.3, whose "
            "mlp/mlp_reg layer layout changed; retrain the model to fix this."
        )


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
