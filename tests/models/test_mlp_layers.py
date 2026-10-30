"""Unit tests for nkululeko.models.mlp_layers: MLPBlock and
parse_drop_config, shared by model_mlp.py (MLPModel) and
model_mlp_regression.py (MLP_Reg_model) -- see test_model_mlp.py for
MLPBlock's layer-order/activation/dropout-placement tests (GH #467 / PR
#476 review), covered once here since both model types now share this
one implementation.
"""

import pytest

from nkululeko.models.mlp_layers import parse_drop_config


class _Util:
    def __init__(self, drop):
        self._drop = drop

    def config_val(self, section, key, default=None):
        assert (section, key) == ("MODEL", "drop")
        return self._drop if self._drop is not None else default

    def error(self, msg):
        raise ValueError(msg)


class TestParseDropConfig:
    """GH #476 review: model_mlp_regression.py read `[MODEL] drop` with
    `self.util.config_val("MODEL", "drop", False)` and passed it straight
    to `float(drop)`. config_val returns the raw ini *string*, so
    `drop = False` in the ini (the string "False", which is truthy) or a
    per-layer list crashed with ValueError before a single batch ran.
    model_mlp.py already parsed it correctly inline; parse_drop_config is
    that same parsing, shared so both model types -- and model_mlp.py's
    own load()/load_path() -- use it identically."""

    def test_default_is_false(self):
        assert parse_drop_config(_Util(None)) is False

    def test_literal_false_string_does_not_crash(self):
        # The exact crash repro: `[MODEL] drop = False` in the ini, read
        # back as the string "False", used to reach float("False").
        assert parse_drop_config(_Util("False")) is False

    def test_parses_float_string(self):
        assert parse_drop_config(_Util("0.5")) == pytest.approx(0.5)

    def test_parses_list_string(self):
        # The other crash repro: a per-layer dropout list, e.g.
        # `[MODEL] drop = [0.3, 0.2]`, used to reach float("[0.3, 0.2]").
        assert parse_drop_config(_Util("[0.3, 0.2]")) == [0.3, 0.2]

    def test_invalid_value_calls_util_error(self):
        with pytest.raises(ValueError):
            parse_drop_config(_Util("not-a-number"))
