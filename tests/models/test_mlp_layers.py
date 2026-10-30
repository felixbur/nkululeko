"""Unit tests for nkululeko.models.mlp_layers.parse_drop_config.
MLPBlock's own tests live in test_model_mlp.py."""

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
    """GH #476 review: model_mlp_regression.py passed the raw ini string
    straight to float(drop), crashing on "False" or a per-layer list."""

    def test_default_is_false(self):
        assert parse_drop_config(_Util(None)) is False

    def test_literal_false_string_does_not_crash(self):
        assert parse_drop_config(_Util("False")) is False

    def test_parses_float_string(self):
        assert parse_drop_config(_Util("0.5")) == pytest.approx(0.5)

    def test_parses_list_string(self):
        assert parse_drop_config(_Util("[0.3, 0.2]")) == [0.3, 0.2]

    def test_invalid_value_calls_util_error(self):
        with pytest.raises(ValueError):
            parse_drop_config(_Util("not-a-number"))
