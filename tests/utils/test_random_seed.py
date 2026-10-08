import pytest

from nkululeko.utils.random_seed import parse_seed


@pytest.mark.parametrize("value", ["False", "false", "None", "", False, "0", 0])
def test_unset_values_give_none(value):
    assert parse_seed(value) is None


@pytest.mark.parametrize("value, expected", [("42", 42), (7, 7), (" 123 ", 123)])
def test_integer_values_are_parsed(value, expected):
    assert parse_seed(value) == expected


@pytest.mark.parametrize("value", ["abc", "1.5", "__import__('os')"])
def test_invalid_values_raise_without_evaluating(value):
    with pytest.raises(ValueError):
        parse_seed(value)
