# mlp_layers.py
"""Shared hidden-layer stack for `mlp`/`mlp_reg` (model_mlp.py /
model_mlp_regression.py): a sequence of Linear/activation/dropout blocks
followed by one output Linear, built from `[MODEL] layers`.

Historically each model file had its own copy of this class, and the two
copies had already drifted (GH #467: a key collision silently dropped one
activation per 2+ hidden layer config in both copies; per-layer dropout
lists only worked in one of them). Sharing one implementation means a fix
here applies to both model types at once.
"""

import ast
from collections import OrderedDict

import torch


def parse_drop_config(util):
    """Parse `[MODEL] drop` into `False`, a float, or a list of floats (one
    per hidden layer). `util.config_val` returns the raw ini string, so a
    plain `float(drop)` crashes on `"False"` or `"[0.3, 0.2]"`.
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
    return drop


class MLPBlock(torch.nn.Module):
    """i -> layers[0] -> ... -> layers[-1] -> o, with `activation` after
    each hidden Linear and, if `drop`, a Dropout after that.

    `layers` is the parsed `[MODEL] layers` value: a list of hidden widths
    (`[64, 16]`) or a dict of them (`{'l1': 64, 'l2': 16}`). Either way the
    widths are used in the order given -- not resorted by width -- so
    `[64, 16]` builds a 64-wide hidden layer then a 16-wide one, same as
    the ini file reads.

    `drop` is `False` (no dropout), a single number (applied after every
    hidden layer), or a list of numbers (applied positionally, one per
    hidden layer; a hidden layer past the end of the list gets none).
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
