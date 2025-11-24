from itertools import combinations
import math

import numpy as np
import pandas as pd
from scipy import stats


def check_na(a):
    if np.isnan(a).any():
        count = np.count_nonzero(np.isnan(a))
        print(f"WARNING: got {count} Nans (of {len(a)}), setting to 0")
        a[np.isnan(a)] = 0
    return a


def cohen_d(d1: np.array, d2: np.array) -> float:
    """Compute Cohen's d from two distributions of real valued arrays.

    Args:
        d1: one array
        d2: the other array
    Returns:
        Cohen's d with precision 3
    """
    # Checks:
    d1 = check_na(d1)
    d2 = check_na(d2)
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    if math.isnan(s) or s == 0:
        return -1
    return (int(1000 * np.abs((u1 - u2)) / s)) / 1000


def all_combinations(items_list):
    result = []
    for i in range(1, len(items_list) + 1):
        for combo in combinations(items_list, i):
            if len(combo) == 2:
                result.append(list(combo))
    return result


def get_effect_size(
    df: pd.DataFrame, target: str, variable: str
) -> tuple[str, str, dict]:
    """Get the effect size as Cohen's D.

    Effect size is computed  from a real numbered variable on a categorical target.

    Args:
        df: a pd.Dataframe with at least target and variable as columns
        target: the categorical target, e.g. emotion
        variable: the real numbered variable that might have an effect, e.g. SNR
    Returns:
        The categories with the maximal Cohen's d based on the variable as a string
        The Cohen's d for this combination
    """
    categories = df[target].unique()
    cats = {}
    for c in categories:
        cats[c] = df[df[target] == c][variable].values
    combos = all_combinations(categories)
    results = {categories[0]: 0}
    if len(categories) == 1:
        cat_s = cohens_D_to_string(0)
        return categories[0], cat_s, results
    else:
        for combo in combos:
            one = combo[0]
            other = combo[1]
            results[f"{one}-{other}"] = cohen_d(cats[one], cats[other])
        max_cat = max(results, key=results.get)
        cat_s = cohens_D_to_string(float(results[max_cat]))
    return max_cat, cat_s, results


def cohens_D_to_string(val: float) -> str:
    if val < 0.2:
        rval = "no effect"
    elif val < 0.2:
        rval = "small effect"
    elif val < 0.5:
        rval = "middle effect"
    else:
        rval = "large effect"
    return f"Cohen's d: {rval}"


def p_value_to_string(p_val: float) -> str:
    """Convert p-value to interpretable significance string.

    Args:
        p_val: p-value from statistical test
    Returns:
        String describing statistical significance level
    """
    if p_val < 0.001:
        return f"highly significant ({p_val:.3f} < 0.001)"
    elif p_val < 0.01:
        return f"significant ({p_val:.3f} < 0.01)"
    elif p_val < 0.05:
        return f"significant ({p_val:.3f}< 0.05)"
    elif p_val < 0.1:
        return f"marginally significant ({p_val:.3f} < 0.1)"
    else:
        return f"not significant ({p_val:.3f} >= 0.1)"


def get_t_test_effect(
    variable1: np.array, variable2: np.array
) -> tuple[float, float, str]:
    """Get t-test statistics for two real-numbered distributions.

    Performs independent samples t-test between two continuous variables.

    Args:
        variable1: first real-numbered distribution
        variable2: second real-numbered distribution
    Returns:
        t-statistic
        p-value
        significance interpretation string
    """

    # Handle NaN values
    variable1 = check_na(variable1)
    variable2 = check_na(variable2)

    # check for equal variance
    _, p_levene = stats.levene(variable1, variable2)
    equal_var = False
    if p_levene > .05:
        equal_var = True
        

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(variable1, variable2, equal_var=equal_var)

    # Create interpretation string
    significance = p_value_to_string(p_value)

    return float(t_stat), float(p_value), significance


def get_mannwhitney_effect(
    variable1: np.array, variable2: np.array
) -> tuple[float, float, str]:
    """Get Mann-Whitney U test statistics for two real-numbered distributions.

    Performs Mann-Whitney U test between two continuous variables.
    This is a non-parametric test that doesn't assume normal distributions.

    Args:
        variable1: first real-numbered distribution
        variable2: second real-numbered distribution
    Returns:
        U-statistic
        p-value
        significance interpretation string
    """
    # Handle NaN values
    variable1 = check_na(variable1)
    variable2 = check_na(variable2)

    # Perform Mann-Whitney U test
    u_stat, p_value = stats.mannwhitneyu(variable1, variable2, alternative="two-sided")

    # Create interpretation string
    significance = p_value_to_string(p_value)

    return float(u_stat), float(p_value), significance

def normaltest(variable1: np.array):
    # This function tests the null hypothesis that a sample comes from a normal distribution.
    res = stats.normaltest(variable1)
    if res.pvalue >= .05:
        return True
    return False



def get_2cont_effect(
    variable1: np.array, variable2: np.array
) -> tuple[float, float, str]:
    """Calculate statistical significance between two continuous variables.

    Automatically selects the appropriate statistical test based on data distribution:
    - Uses t-test if both variables are normally distributed and sample size > 30
    - Otherwise uses Mann-Whitney U test (non-parametric alternative)

    Args:
        variable1: First continuous variable array
        variable2: Second continuous variable array

    Returns:
        Dictionary containing:
            - approach: Statistical test used ("t-test" or "mann-whitney")
            - significance: Human-readable significance level string
            - p-val: Raw p-value from the test
    """
    if normaltest(variable1) and normaltest(variable2) and len(variable1) > 30:
        _, p_value, significance = get_t_test_effect(variable1, variable2)
        approach = "t-test"
    else:
        _, p_value, significance = get_mannwhitney_effect(variable1, variable2)
        approach = "mann-whitney"
    significance = p_value_to_string(p_value)
    return {"approach":approach, "significance":significance, "p-val":p_value}

def find_most_significant_difference_ttests(
    distributions: dict
) -> tuple[str, float, float, str, dict]:
    """Find the combination with the most significant t-test difference among n distributions.

    Args:
        distributions: dictionary with distribution names as keys and numpy arrays as values
        equal_var: whether to assume equal variances (default True)
                  If False, uses Welch's t-test
    Returns:
        Most significant pair as string "dist1-dist2"
        t-statistic for most significant pair
        p-value for most significant pair
        significance interpretation for most significant pair
        Dictionary with all pairwise comparisons containing t_stat, p_value, significance
    """
    if len(distributions) < 2:
        raise ValueError("Need at least 2 distributions for comparison")

    dist_names = list(distributions.keys())
    combos = all_combinations(dist_names)
    results = {}

    for combo in combos:
        name1, name2 = combo[0], combo[1]
        dist1, dist2 = distributions[name1], distributions[name2]

        t_stat, p_value, significance = get_t_test_effect(dist1, dist2)

        combo_key = f"{name1}-{name2}"
        results[combo_key] = {
            "t_stat": t_stat,
            "p_value": p_value,
            "significance": significance,
        }

    # Find combination with most significant p-value (smallest p-value)
    min_p_key = min(results, key=lambda x: results[x]["p_value"])
    min_p_result = results[min_p_key]

    return {
        "approach": "t-test",
        "combo": min_p_key,
        "t_stat": min_p_result["t_stat"],
        "p_value": min_p_result["p_value"],
        "significance": min_p_result["significance"],
        "all_results": results,
    }


def get_kruskal_wallis_effect(distributions: dict) -> tuple[float, float, str]:
    """Get Kruskal-Wallis test statistics for multiple real-numbered distributions.

    Performs Kruskal-Wallis test among multiple continuous variables.
    This is a non-parametric test equivalent to one-way ANOVA.

    Args:
        distributions: dictionary with distribution names as keys and numpy arrays as values
    Returns:
        H-statistic (Kruskal-Wallis statistic)
        p-value
        significance interpretation string
    """
    if len(distributions) < 3:
        raise ValueError("Kruskal-Wallis test requires at least 3 distributions")

    # Handle NaN values for all distributions
    cleaned_dists = []
    for dist in distributions.values():
        cleaned_dists.append(check_na(dist))

    # Perform Kruskal-Wallis test
    h_stat, p_value = stats.kruskal(*cleaned_dists)

    # Create interpretation string
    significance = p_value_to_string(p_value)

    return float(h_stat), float(p_value), significance


def find_most_significant_difference(
    distributions: dict, mean_featnum: float = 0
) -> tuple[str, dict, dict]:
    """Find the most significant difference between multiple distributions.

    Automatically selects the appropriate statistical test based on the number
    of distributions and mean feature number:
    - For 2 distributions with mean_featnum >= 30: t-test (Welch's if unequal var)
    - For 2 distributions with mean_featnum < 30: Mann-Whitney U test
    - For >2 distributions: Kruskal-Wallis test (non-parametric ANOVA)

    Args:
        distributions (dict): Dictionary with distribution names as keys and
                             numpy arrays as values
        mean_featnum (float): Mean number of features/samples per distribution.
                             Determines whether to use parametric (>=30) or
                             non-parametric (<30) tests. Defaults to 0.

    Returns:
        tuple: (pairwise_results, overall_results) where:
            - pairwise_results (dict): Results for pairwise comparisons with keys:
                'approach', 'combo', test statistic, 'p_value', 'significance',
                'all_results'
            - overall_results (dict): Results for overall test (Kruskal-Wallis if >2
                distributions), or None if only 2 distributions

    Raises:
        ValueError: If fewer than 2 distributions are provided

    Examples:
    --------
    >>> distributions = {
    ...     'group_A': np.array([1.2, 1.5, 1.8, 2.1]),
    ...     'group_B': np.array([2.3, 2.7, 3.1, 3.4]),
    ...     'group_C': np.array([3.8, 4.1, 4.5, 4.9])
    ... }
    >>> approach, pairwise, overall = find_most_significant_difference(
    ...     distributions, mean_featnum=25)
    >>> # Returns Mann-Whitney U for pairwise, Kruskal-Wallis for overall
    """
    if len(distributions) < 2:
        raise ValueError("Need at least 2 distributions for comparison")
    results_bin = None
    res_all = None
    if mean_featnum >= 30:
        results_bin = find_most_significant_difference_ttests(
            distributions
        )
    else:
        results_bin = find_most_significant_difference_mannwhitney(distributions)

    if len(distributions) > 2:
        # Use Kruskal-Wallis test for >2 distributions
        h_stat, p_value, significance = get_kruskal_wallis_effect(distributions)

        results_kruskal_wallis = {
            "all_groups": {
                "h_stat": h_stat,
                "p_value": p_value,
                "significance": significance,
            }
        }
        approach = "Kruskal-Wallis"
        res_all = {
            "approach": approach,
            "combo": "all_groups",
            "h_stat": h_stat,
            "p_value": p_value,
            "significance": significance,
            "all_results": results_kruskal_wallis,
        }

    return results_bin, res_all


def find_most_significant_difference_mannwhitney(distributions: dict) -> dict:
    """Find the most significant difference using Mann-Whitney U tests.

    Performs pairwise Mann-Whitney U tests between all distributions and
    returns the combination with the most significant p-value. This is a
    non-parametric alternative that doesn't assume normal distributions.

    Args:
        distributions (dict): Dictionary with distribution names as keys and
                             numpy arrays as values

    Returns:
        dict: Results dictionary with keys:
            - 'approach': Always 'Mann-Whitney U'
            - 'combo': Most significant pair as string "dist1-dist2"
            - 'u_stat': U-statistic for most significant pair
            - 'p_value': p-value for most significant pair
            - 'significance': significance interpretation string
            - 'all_results': Dictionary with all pairwise comparisons

    Raises:
        ValueError: If fewer than 2 distributions are provided

    Examples:
    --------
    >>> distributions = {
    ...     'group_A': np.array([1.2, 1.5, 1.8, 2.1]),
    ...     'group_B': np.array([2.3, 2.7, 3.1, 3.4]),
    ...     'group_C': np.array([3.8, 4.1, 4.5, 4.9])
    ... }
    >>> result = find_most_significant_difference_mannwhitney(distributions)
    >>> print(f"Most significant: {result['combo']} (p={result['p_value']:.3f})")
    """
    if len(distributions) < 2:
        raise ValueError("Need at least 2 distributions for comparison")

    dist_names = list(distributions.keys())
    combos = all_combinations(dist_names)
    results = {}

    for combo in combos:
        name1, name2 = combo[0], combo[1]
        dist1, dist2 = distributions[name1], distributions[name2]

        u_stat, p_value, significance = get_mannwhitney_effect(dist1, dist2)

        combo_key = f"{name1}-{name2}"
        results[combo_key] = {
            "u_stat": u_stat,
            "p_value": p_value,
            "significance": significance,
        }

    # Find combination with most significant p-value (smallest p-value)
    min_p_key = min(results, key=lambda x: results[x]["p_value"])
    min_p_result = results[min_p_key]

    return {
        "approach": "Mann-Whitney U",
        "combo": min_p_key,
        "u_stat": min_p_result["u_stat"],
        "p_value": min_p_result["p_value"],
        "significance": min_p_result["significance"],
        "all_results": results,
    }


def normalize(values):
    """Do a z-transformation of a distribution.

    So that mean = 0 and variance = 1
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    return scaler.fit_transform(values)
