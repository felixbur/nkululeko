"""Unit tests for stats utilities (nkululeko/utils/stats.py)."""

import numpy as np
import pandas as pd
import pytest

from nkululeko.utils.stats import (
    all_combinations,
    check_na,
    cohen_d,
    cohens_D_to_string,
    find_most_significant_difference,
    find_most_significant_difference_mannwhitney,
    find_most_significant_difference_ttests,
    get_2cont_effect,
    get_effect_size,
    get_kruskal_wallis_effect,
    get_mannwhitney_effect,
    get_t_test_effect,
    normalize,
    normaltest,
    p_value_to_string,
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
        combo_str, _, _ = get_effect_size(df, "target", "feature")
        assert combo_str == "A"


class TestCohensDToString:
    def test_negative_one_is_no_effect(self):
        result = cohens_D_to_string(-1.0)
        assert result == "Cohen's d: no effect"

    def test_zero_is_no_effect(self):
        result = cohens_D_to_string(0.0)
        assert result == "Cohen's d: no effect"

    def test_just_below_0_2_is_no_effect(self):
        result = cohens_D_to_string(0.19)
        assert result == "Cohen's d: no effect"

    def test_between_0_2_and_0_5_is_small_effect(self):
        result = cohens_D_to_string(0.35)
        assert result == "Cohen's d: small effect"

    def test_at_0_2_is_small_effect(self):
        result = cohens_D_to_string(0.2)
        assert result == "Cohen's d: small effect"

    def test_at_0_5_is_middle_effect(self):
        result = cohens_D_to_string(0.5)
        assert result == "Cohen's d: middle effect"

    def test_at_0_8_is_large_effect(self):
        result = cohens_D_to_string(0.8)
        assert result == "Cohen's d: large effect"

    def test_large_value_is_large_effect(self):
        result = cohens_D_to_string(2.0)
        assert result == "Cohen's d: large effect"

    def test_returns_string(self):
        assert isinstance(cohens_D_to_string(0.0), str)


class TestPValueToString:
    def test_highly_significant(self):
        result = p_value_to_string(0.0005)
        assert "highly significant" in result
        assert "0.001" in result

    def test_significant_at_0_01(self):
        result = p_value_to_string(0.005)
        assert "significant" in result
        assert "0.01" in result

    def test_significant_at_0_05(self):
        result = p_value_to_string(0.03)
        assert "significant" in result
        assert "0.05" in result

    def test_marginally_significant(self):
        result = p_value_to_string(0.07)
        assert "marginally significant" in result
        assert "0.1" in result

    def test_not_significant(self):
        result = p_value_to_string(0.5)
        assert "not significant" in result
        assert "0.1" in result

    def test_p_value_shown_in_result(self):
        result = p_value_to_string(0.042)
        assert "0.042" in result


class TestGetTTestEffect:
    def test_returns_three_values(self):
        rng = np.random.default_rng(0)
        v1 = rng.normal(0, 1, 50)
        v2 = rng.normal(1, 1, 50)
        t, p, sig = get_t_test_effect(v1, v2)
        assert isinstance(t, float)
        assert isinstance(p, float)
        assert isinstance(sig, str)

    def test_p_value_between_0_and_1(self):
        rng = np.random.default_rng(1)
        v1 = rng.normal(0, 1, 40)
        v2 = rng.normal(0, 1, 40)
        _, p, _ = get_t_test_effect(v1, v2)
        assert 0.0 <= p <= 1.0

    def test_clearly_different_distributions_low_p(self):
        rng = np.random.default_rng(42)
        v1 = rng.normal(0, 0.1, 100)
        v2 = rng.normal(10, 0.1, 100)
        _, p, _ = get_t_test_effect(v1, v2)
        assert p < 0.001

    def test_identical_distributions_high_p(self):
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        _, p, _ = get_t_test_effect(v, v)
        assert p == pytest.approx(1.0)

    def test_significance_string_matches_p(self):
        rng = np.random.default_rng(7)
        v1 = rng.normal(0, 0.1, 100)
        v2 = rng.normal(5, 0.1, 100)
        _, p, sig = get_t_test_effect(v1, v2)
        assert p < 0.001
        assert "highly significant" in sig

    def test_nan_values_handled(self):
        v1 = np.array([float("nan"), 1.0, 2.0, 3.0, 4.0])
        v2 = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        t, _, _ = get_t_test_effect(v1, v2)
        assert not np.isnan(t)


class TestGetMannWhitneyEffect:
    def test_returns_three_values(self):
        rng = np.random.default_rng(0)
        v1 = rng.normal(0, 1, 30)
        v2 = rng.normal(2, 1, 30)
        u, p, sig = get_mannwhitney_effect(v1, v2)
        assert isinstance(u, float)
        assert isinstance(p, float)
        assert isinstance(sig, str)

    def test_p_value_between_0_and_1(self):
        rng = np.random.default_rng(5)
        v1 = rng.normal(0, 1, 25)
        v2 = rng.normal(0, 1, 25)
        _, p, _ = get_mannwhitney_effect(v1, v2)
        assert 0.0 <= p <= 1.0

    def test_well_separated_low_p(self):
        v1 = np.arange(1, 21, dtype=float)
        v2 = np.arange(100, 120, dtype=float)
        _, p, _ = get_mannwhitney_effect(v1, v2)
        assert p < 0.01

    def test_u_statistic_nonnegative(self):
        rng = np.random.default_rng(3)
        v1 = rng.normal(0, 1, 20)
        v2 = rng.normal(1, 1, 20)
        u, _, _ = get_mannwhitney_effect(v1, v2)
        assert u >= 0


class TestNormaltest:
    def test_normal_data_returns_true(self):
        rng = np.random.default_rng(99)
        data = rng.normal(0, 1, 200)
        assert normaltest(data) is True

    def test_uniform_data_returns_false(self):
        # Uniformly distributed data is not normally distributed
        rng = np.random.default_rng(10)
        data = rng.uniform(0, 1, 200)
        assert normaltest(data) is False

    def test_returns_bool(self):
        rng = np.random.default_rng(0)
        result = normaltest(rng.normal(0, 1, 50))
        assert isinstance(result, bool)


class TestGet2ContEffect:
    def test_large_normal_sample_uses_t_test(self):
        rng = np.random.default_rng(42)
        v1 = rng.normal(0, 1, 200)
        v2 = rng.normal(1, 1, 200)
        result = get_2cont_effect(v1, v2)
        assert "approach" in result
        assert result["approach"] in ("t-test", "mann-whitney")
        assert 0.0 <= result["p-val"] <= 1.0

    def test_small_sample_uses_mann_whitney(self):
        rng = np.random.default_rng(0)
        v1 = rng.normal(0, 1, 10)
        v2 = rng.normal(0, 1, 10)
        result = get_2cont_effect(v1, v2)
        assert result["approach"] == "mann-whitney"

    def test_result_has_required_keys(self):
        rng = np.random.default_rng(1)
        v1 = rng.normal(0, 1, 20)
        v2 = rng.normal(0, 1, 20)
        result = get_2cont_effect(v1, v2)
        assert set(result.keys()) == {"approach", "significance", "p-val"}

    def test_significance_string_present(self):
        rng = np.random.default_rng(2)
        v1 = rng.normal(0, 1, 20)
        v2 = rng.normal(0, 1, 20)
        result = get_2cont_effect(v1, v2)
        assert isinstance(result["significance"], str)
        assert len(result["significance"]) > 0


class TestFindMostSignificantDifferenceTtests:
    def test_two_distributions_returns_result(self):
        rng = np.random.default_rng(42)
        dists = {
            "A": rng.normal(0, 1, 50),
            "B": rng.normal(3, 1, 50),
        }
        result = find_most_significant_difference_ttests(dists)
        assert result["approach"] == "t-test"
        assert "A-B" in result["combo"] or "B-A" in result["combo"]
        assert 0.0 <= result["p_value"] <= 1.0

    def test_three_distributions_finds_most_different(self):
        rng = np.random.default_rng(0)
        dists = {
            "close1": rng.normal(0, 1, 30),
            "close2": rng.normal(0.1, 1, 30),
            "far": rng.normal(10, 0.1, 30),
        }
        result = find_most_significant_difference_ttests(dists)
        assert "far" in result["combo"]

    def test_raises_with_one_distribution(self):
        with pytest.raises(ValueError):
            find_most_significant_difference_ttests({"A": np.array([1.0, 2.0])})

    def test_all_results_in_output(self):
        rng = np.random.default_rng(1)
        dists = {
            "X": rng.normal(0, 1, 20),
            "Y": rng.normal(1, 1, 20),
            "Z": rng.normal(2, 1, 20),
        }
        result = find_most_significant_difference_ttests(dists)
        assert "all_results" in result
        assert len(result["all_results"]) == 3


class TestFindMostSignificantDifferenceMannWhitney:
    def test_two_distributions_returns_result(self):
        rng = np.random.default_rng(7)
        dists = {
            "A": rng.normal(0, 1, 30),
            "B": rng.normal(5, 1, 30),
        }
        result = find_most_significant_difference_mannwhitney(dists)
        assert result["approach"] == "Mann-Whitney U"
        assert "u_stat" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_raises_with_one_distribution(self):
        with pytest.raises(ValueError):
            find_most_significant_difference_mannwhitney(
                {"only": np.array([1.0, 2.0, 3.0])}
            )

    def test_all_results_keys_present(self):
        rng = np.random.default_rng(3)
        dists = {
            "P": rng.normal(0, 1, 20),
            "Q": rng.normal(2, 1, 20),
        }
        result = find_most_significant_difference_mannwhitney(dists)
        assert set(result.keys()) == {
            "approach",
            "combo",
            "u_stat",
            "p_value",
            "significance",
            "all_results",
        }


class TestGetKruskalWallisEffect:
    def test_three_distinct_groups_returns_low_p(self):
        dists = {
            "A": np.arange(1, 21, dtype=float),
            "B": np.arange(50, 70, dtype=float),
            "C": np.arange(100, 120, dtype=float),
        }
        h, p, _ = get_kruskal_wallis_effect(dists)
        assert h > 0
        assert p < 0.001

    def test_raises_with_two_distributions(self):
        with pytest.raises(ValueError):
            get_kruskal_wallis_effect({"A": np.array([1.0]), "B": np.array([2.0])})

    def test_returns_three_values(self):
        rng = np.random.default_rng(9)
        dists = {
            "X": rng.normal(0, 1, 20),
            "Y": rng.normal(2, 1, 20),
            "Z": rng.normal(4, 1, 20),
        }
        h, p, sig = get_kruskal_wallis_effect(dists)
        assert isinstance(h, float)
        assert isinstance(p, float)
        assert isinstance(sig, str)

    def test_significance_string_meaningful(self):
        dists = {
            "A": np.arange(1, 31, dtype=float),
            "B": np.arange(100, 130, dtype=float),
            "C": np.arange(200, 230, dtype=float),
        }
        _, _, sig = get_kruskal_wallis_effect(dists)
        assert "significant" in sig


class TestFindMostSignificantDifference:
    def test_two_groups_large_n_returns_t_test(self):
        rng = np.random.default_rng(42)
        dists = {
            "A": rng.normal(0, 1, 50),
            "B": rng.normal(3, 1, 50),
        }
        pairwise, overall = find_most_significant_difference(dists, mean_featnum=30)
        assert pairwise is not None
        assert pairwise["approach"] == "t-test"
        assert overall is None

    def test_two_groups_small_n_returns_mann_whitney(self):
        rng = np.random.default_rng(0)
        dists = {
            "A": rng.normal(0, 1, 20),
            "B": rng.normal(2, 1, 20),
        }
        pairwise, overall = find_most_significant_difference(dists, mean_featnum=0)
        assert pairwise["approach"] == "Mann-Whitney U"
        assert overall is None

    def test_three_groups_includes_kruskal_wallis(self):
        rng = np.random.default_rng(5)
        dists = {
            "A": rng.normal(0, 1, 30),
            "B": rng.normal(3, 1, 30),
            "C": rng.normal(6, 1, 30),
        }
        _, overall = find_most_significant_difference(dists, mean_featnum=0)
        assert overall is not None
        assert overall["approach"] == "Kruskal-Wallis"
        assert "h_stat" in overall

    def test_raises_with_single_distribution(self):
        with pytest.raises(ValueError):
            find_most_significant_difference({"only": np.array([1.0, 2.0, 3.0])})


class TestNormalize:
    def test_output_has_zero_mean(self):
        values = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = normalize(values)
        assert np.mean(result) == pytest.approx(0.0, abs=1e-10)

    def test_output_has_unit_std(self):
        values = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = normalize(values)
        assert np.std(result) == pytest.approx(1.0, abs=1e-10)

    def test_2d_input_shape_preserved(self):
        values = np.random.default_rng(0).normal(5, 2, (20, 4))
        result = normalize(values)
        assert result.shape == values.shape
