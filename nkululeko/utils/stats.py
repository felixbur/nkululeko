from itertools import combinations
import math
import numpy as np
import pandas as pd


def check_na(a):
    if np.isnan(a).any():
        count = np.count_nonzero(np.isnan(a))
        print(f"WARNING: got {count} Nans (of {len(a)}), setting to 0")
        a[np.isnan(a)] = 0
        return a
    else:
        return a


def cohen_d(d1, d2):
    """
    Compute Cohen's d from two distributions of real valued arrays
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


def get_effect_size(df, target, variable):
    """
    Get the effect size as Cohen's D from a real numbered variable on a categorical target.
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
    results = {}
    for combo in combos:
        one = combo[0]
        other = combo[1]
        results[f"{one}-{other}"] = cohen_d(cats[one], cats[other])
    max_cat = max(results, key=results.get)
    cat_s = cohens_D_to_string(float(results[max_cat]))
    return max_cat, cat_s, results[max_cat]


def cohens_D_to_string(val):
    if val < 0.2:
        rval = "no effect"
    elif val < 0.2:
        rval = "small effect"
    elif val < 0.5:
        rval = "middle effect"
    else:
        rval = "large effect"
    return f"Cohen's d: {rval}"


def normalize(values):
    """Do a z-transformation of a distribution, so that mean = 0 and variance = 1"""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    return scaler.fit_transform(values)
