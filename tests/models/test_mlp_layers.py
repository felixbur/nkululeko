"""Unit tests for nkululeko.models.mlp_layers.parse_drop_config.
MLPBlock's own tests live in test_model_mlp.py."""

import pytest
import torch

from nkululeko.models.mlp_layers import MLPBlock, load_mlp_state_dict, parse_drop_config


class _Util:
    def __init__(self, drop):
        self._drop = drop
        self.warnings = []

    def config_val(self, section, key, default=None):
        assert (section, key) == ("MODEL", "drop")
        return self._drop if self._drop is not None else default

    def error(self, msg):
        raise ValueError(msg)

    def warn(self, msg):
        self.warnings.append(msg)


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


class TestParseDropConfigValidation:
    """GH #480: drop as a tuple, a bare bool, or an out-of-range/mismatched
    value either crashed uncontrolled or was silently accepted."""

    def test_parses_tuple_string_as_list(self):
        assert parse_drop_config(_Util("(0.3, 0.2)")) == [0.3, 0.2]

    def test_bool_true_is_rejected(self):
        with pytest.raises(ValueError):
            parse_drop_config(_Util("True"))

    def test_float_out_of_range_is_rejected(self):
        with pytest.raises(ValueError):
            parse_drop_config(_Util("1.5"))

    def test_negative_float_is_rejected(self):
        with pytest.raises(ValueError):
            parse_drop_config(_Util("-0.1"))

    def test_list_value_out_of_range_is_rejected(self):
        with pytest.raises(ValueError):
            parse_drop_config(_Util("[0.3, 1.5]"))

    def test_list_length_mismatch_warns(self):
        util = _Util("[0.3, 0.2, 0.1]")
        parse_drop_config(util, num_layers=2)
        assert len(util.warnings) == 1

    def test_list_length_match_does_not_warn(self):
        util = _Util("[0.3, 0.2]")
        parse_drop_config(util, num_layers=2)
        assert util.warnings == []


class TestLoadMlpStateDict:
    """GH #480: load_state_dict() on a pre-1.14.3 checkpoint (different key
    layout) raised a raw RuntimeError; a missing file a raw FileNotFoundError."""

    def test_missing_file_calls_util_error(self):
        with pytest.raises(ValueError):
            load_mlp_state_dict(
                MLPBlock(3, [4], 2, False, torch.nn.ReLU()),
                "/nonexistent/path/model.pt",
                _Util(None),
            )

    def test_key_mismatch_calls_util_error(self, tmp_path):
        path = tmp_path / "model.pt"
        old_model = MLPBlock(3, [4, 2], 2, False, torch.nn.ReLU())
        torch.save(old_model.state_dict(), str(path))
        # a differently-shaped net -> different state_dict keys
        new_model = MLPBlock(3, [5], 2, False, torch.nn.ReLU())

        with pytest.raises(ValueError):
            load_mlp_state_dict(new_model, str(path), _Util(None))

    def test_successful_load(self, tmp_path):
        path = tmp_path / "model.pt"
        model = MLPBlock(3, [4], 2, False, torch.nn.ReLU())
        torch.save(model.state_dict(), str(path))
        other = MLPBlock(3, [4], 2, False, torch.nn.ReLU())

        load_mlp_state_dict(other, str(path), _Util(None))
