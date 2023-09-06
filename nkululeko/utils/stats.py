from itertools import combinations
import math
import numpy as np
import pandas as pd

def cohen_d(d1, d2):
    """
    Compute Cohen's d from two distributions of real valued arrays
    Args:
        d1: one array
        d2: the other array
    Returns:
        Cohen's d with precision 3
    """
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    if math.isnan(s):
        return 0
    return (int(1000*np.abs((u1 - u2)) / s))/1000
 

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
        cats[c] = df[df[target]==c][variable].values
    combos = all_combinations(categories)
    results = {}
    for combo in combos:
        one = combo[0]
        other = combo[1]
        results[f'{one}-{other}'] = cohen_d(cats[one], cats[other])
    max_cat = max(results, key=results.get)
    return max_cat, results[max_cat]