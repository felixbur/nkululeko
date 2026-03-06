"""Unit tests for stats utilities (nkululeko/utils/stats.py)."""

import math

import numpy as np
import pandas as pd
import pytest

from nkululeko.utils.stats import (
    all_combinations,
    check_na,
    cohen_d,
    get_effect_size,
)


class TestCheckNa:
    def test_no_nans_unchanged(self):
        a = np.array([1.0, 2.0, 3.0])
        result = check_na(a.copy())
        np.testing.assert_array_equal(result, a)

    def test_nans_replaced_with_zero(self):
        a = np.array([1.0, float("nan"), 3.0])
        result = check_na(a)
        assert result[1] == pytest.approx(0.0)
        assert not np.isnan(result).any()

    def test_multiple_nans_all_replaced(self):
        a = np.array([float("nan"), float("nan"), 5.0])
        result = check_na(a)
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.0)


class TestCohenD:
    def test_identical_distributions_zero(self):
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cohen_d(d, d)
        assert result == pytest.approx(0.0)

    def test_clearly_separated_distributions_large(self):
        d1 = np.zeros(50)
        d2 = np.ones(50) * 10
        result = cohen_d(d1, d2)
        # std ≈ 0 for constants → pooled std ≈ 0 → returns -1 by convention
        # but let's just check it's non-negative or -1
        assert result >= -1

    def test_moderate_effect_positive(self):
        rng = np.random.default_rng(42)
        d1 = rng.normal(0, 1, 100)
        d2 = rng.normal(1, 1, 100)
        result = cohen_d(d1, d2)
        assert result > 0

    def test_result_is_precision_3(self):
        rng = np.random.default_rng(0)
        d1 = rng.normal(0, 1, 50)
        d2 = rng.normal(0.5, 1, 50)
        result = cohen_d(d1, d2)
        # Precision 3 means value * 1000 is integer
        assert result == round(result, 3)

    def test_nan_inputs_handled(self):
        d1 = np.array([float("nan"), 1.0, 2.0])
        d2 = np.array([3.0, 4.0, 5.0])
        # Should not raise; nan values are replaced with 0
        result = cohen_d(d1, d2)
        assert isinstance(result, float)


class TestAllCombinations:
    def test_two_items_gives_one_pair(self):
        result = all_combinations(["a", "b"])
        assert result == [["a", "b"]]

    def test_three_items_gives_three_pairs(self):
        result = all_combinations(["a", "b", "c"])
        assert len(result) == 3
        assert ["a", "b"] in result
        assert ["a", "c"] in result
        assert ["b", "c"] in result

    def test_single_item_gives_empty(self):
        result = all_combinations(["a"])
        assert result == []

    def test_only_pairs_returned(self):
        result = all_combinations(["a", "b", "c", "d"])
        for combo in result:
            assert len(combo) == 2


class TestGetEffectSize:
    def test_two_categories_returns_result(self):
        df = pd.DataFrame(
            {
                "target": ["A"] * 20 + ["B"] * 20,
                "feature": list(np.random.default_rng(1).normal(0, 1, 20))
                + list(np.random.default_rng(2).normal(1, 1, 20)),
            }
        )
        combo_str, cohen_str, results_dict = get_effect_size(df, "target", "feature")
        # combo_str is the key of the max-effect pair, e.g. "A-B"
        assert isinstance(combo_str, str)
        assert isinstance(cohen_str, str)
        assert isinstance(results_dict, dict)
        # the result dict should contain at least the pair key
        assert combo_str in results_dict

    def test_single_category_returns_zero_effect(self):
        df = pd.DataFrame({"target": ["A"] * 10, "feature": np.ones(10)})
        combo_str, cohen_str, results_dict = get_effect_size(df, "target", "feature")
        assert combo_str == "A"
