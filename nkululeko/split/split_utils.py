"""
Code copyright by Uwe Reichel
"""

from collections import Counter
import numpy as np
import pandas as pd
import scipy.spatial as ssp
from sklearn.model_selection import GroupShuffleSplit
import sys


def optimize_traindevtest_split(
    X, y, split_on, stratify_on, weight=None, dev_size=0.1, test_size=0.1, k=30, seed=42
):
    """optimize group-disjunct split into training, dev, and test set, which is guided by:
    - disjunct split of values in SPLIT_ON
    - stratification by all keys in STRATIFY_ON (targets and groupings)
    - test set proportion in X should be close to test_size (which is the test
      proportion in set(split_on))

    Score to be minimized: (sum_v[w(v) * max_irad(v)] + w(d) * max_d) / (sum_v[w(v)] + w(d))
    (v: variables to be stratified on
    w(v): their weight
    max_irad(v): maximum information radius of reference distribution of classes in v and
                 - dev set distribution,
                 - test set distribution
    N(v): number of stratification variables
    max_d: maximum of absolute difference between dev and test sizes of X and set(split_on)
    w(d): its weight

    Args:
    X: (pd.DataFrame) of features/groupings for which best split
      is to be calculated. Of shape (N, M)
    y: (np.array) of targets of length N
      if type(y[0]) in ["str", "int"]: y is assumed to be categorical, so that it is additionally
      tested that all partitions cover all classes. Else y is assumed to be numeric and no
      coverage test is done.
    split_on: (np.array) list of length N with grouping variable (e.g. speaker IDs),
      on which the group-disjunct split is to be performed. Must be categorical.
    stratify_on: (dict) Dict-keys are variable names (targets and/or further groupings)
      the split should be stratified on (groupings could e.g. be sex, age class, etc).
      Dict-Values are np.array-s of length N that contain the variable values. All
      variables must be categorical.
    weight: (dict) weight for each variable in stratify_on. Defines their amount of
      contribution to the optimization score. Uniform weighting by default. Additional
      key: "size_diff" defines how the corresponding size differences should be weighted.
    dev_size: (float) proportion in set(split_on) for dev set, e.g. 10% of speakers
      to be held-out
    test_size: (float) test proportion in set(split_on) for test set
    k: (int) number of different splits to be tried out
    seed: (int) random seed
    Returns:
    train_i: (np.array) train set indices in X
    dev_i: (np.array) dev set indices in X
    test_i: (np.array) test set indices in X
    info: (dict) detail information about reference and achieved prob distributions
        "dev_size_in_spliton": intended grouping dev_size
        "dev_size_in_X": optimized dev proportion of observations in X
        "test_size_in_spliton": intended grouping test_size
        "test_size_in_X": optimized test proportion of observations in X
        "p_ref_{c}": reference class distribution calculated from stratify_on[c]
        "p_dev_{c}": dev set class distribution calculated from stratify_on[c][dev_i]
        "p_test_{c}": test set class distribution calculated from stratify_on[c][test_i]
    """

    # data size
    N = len(y)

    # categorical target: number of classes for coverage test
    if is_categorical(y[0]):
        nc = len(set(y))
    else:
        nc = None

    # adjusted dev_size after having split off the test set
    dev_size_adj = (dev_size * N) / (N - test_size * N)

    # split all into train/dev vs test
    gss_o = GroupShuffleSplit(n_splits=k, test_size=test_size, random_state=seed)

    # split train/dev into train vs dev
    gss_i = GroupShuffleSplit(n_splits=k, test_size=dev_size_adj, random_state=seed)

    # set weight defaults
    if weight is None:
        weight = {}
    for c in stratify_on.keys():
        if c not in weight:
            weight[c] = 1
    if "size_diff" not in weight:
        weight["size_diff"] = 1

    # stratification reference distributions calculated on stratify_on
    p_ref = {}
    for c in stratify_on:
        p_ref[c] = class_prob(stratify_on[c])

    # best train/dev/test indices in X; best associated score
    train_i, dev_i, test_i, best_sco = None, None, None, np.inf

    # full target coverage in all partitions
    full_target_coverage = False

    # brute-force optimization of SPLIT_ON split
    #    outer loop *_o: splitting into train/dev and test
    #    inner loop *_i: spltting into train and dev
    for tri_o, tei_o in gss_o.split(X, y, split_on):
        # current train/dev partition
        X_i = X.iloc[tri_o]
        y_i = y[tri_o]
        split_on_i = split_on[tri_o]

        for tri_i, tei_i in gss_i.split(X_i, y_i, split_on_i):
            # all classes maintained in all partitions?
            if nc:
                nc_train = len(set(y[tri_o[tri_i]]))
                nc_dev = len(set(y[tri_o[tei_i]]))
                nc_test = len(set(y[tei_o]))
                if min(nc_train, nc_dev, nc_test) < nc:
                    continue

            full_target_coverage = True

            sco = calc_split_score(
                test_i=tei_o,
                stratify_on=stratify_on,
                weight=weight,
                p_ref=p_ref,
                N=N,
                test_size=test_size,
                dev_i=tri_o[tei_i],
                dev_size=dev_size_adj,
            )

            if sco < best_sco:
                best_sco = sco
                test_i = tei_o
                train_i = tri_o[tri_i]
                dev_i = tri_o[tei_i]

    if test_i is None:
        sys.exit(exit_message(full_target_coverage, "dev and test"))

    # matching info
    info = {
        "score": best_sco,
        "size_devset_in_spliton": dev_size,
        "size_devset_in_X": np.round(len(dev_i) / N, 2),
        "size_testset_in_spliton": test_size,
        "size_testset_in_X": np.round(len(test_i) / N, 2),
    }

    for c in p_ref:
        info[f"p_{c}_ref"] = p_ref[c]
        info[f"p_{c}_dev"] = class_prob(stratify_on[c][dev_i])
        info[f"p_{c}_test"] = class_prob(stratify_on[c][test_i])

    return train_i, dev_i, test_i, info


def optimize_traintest_split(
    X, y, split_on, stratify_on, weight=None, test_size=0.1, k=30, seed=42
):
    """optimize group-disjunct split which is guided by:
    - disjunct split of values in SPLIT_ON
    - stratification by all keys in STRATIFY_ON (targets and groupings)
    - test set proportion in X should be close to test_size (which is the test
      proportion in set(split_on))

    Score to be minimized: (sum_v[w(v) * irad(v)] + w(d) * d) / (sum_v[w(v)] + w(d))
    (v: variables to be stratified on
    w(v): their weight
    irad(v): information radius between reference distribution of classes in v
        and test set distribution
    N(v): number of stratification variables
    d: absolute difference between test sizes of X and set(split_on)
    w(d): its weight

    Args:
    X: (pd.DataFrame) of features/groupings for which best split
      is to be calculated. Of shape (N, M)
    y: (np.array) of targets of length N
      if type(y[0]) in ["str", "int"]: y is assumed to be categorical, so that it is additionally
      tested that all partitions cover all classes. Else y is assumed to be numeric and no
      coverage test is done.
    split_on: (np.array) list of length N with grouping variable (e.g. speaker IDs),
      on which the group-disjunct split is to be performed. Must be categorical.
    stratify_on: (dict) Dict-keys are variable names (targets and/or further groupings)
      the split should be stratified on (groupings could e.g. be sex, age class, etc).
      Dict-Values are np.array-s of length N that contain the variable values. All
      variables must be categorical.
    weight: (dict) weight for each variable in stratify_on. Defines their amount of
      contribution to the optimization score. Uniform weighting by default. Additional
      key: "size_diff" defines how test size diff should be weighted.
    test_size: (float) test proportion in set(split_on), e.g. 10% of speakers to be held-out
    k: (int) number of different splits to be tried out
    seed: (int) random seed
    Returns:
    train_i: (np.array) train set indices in X
    test_i: (np.array) test set indices in X
    info: (dict) detail information about reference and achieved prob distributions
        "size_testset_in_spliton": intended test_size
        "size_testset_in_X": optimized test proportion in X
        "p_ref_{c}": reference class distribution calculated from stratify_on[c]
        "p_test_{c}": test set class distribution calculated from stratify_on[c][test_i]
    """

    gss = GroupShuffleSplit(n_splits=k, test_size=test_size, random_state=seed)

    # set weight defaults
    if weight is None:
        weight = {}
    for c in stratify_on.keys():
        if c not in weight:
            weight[c] = 1
    if "size_diff" not in weight:
        weight["size_diff"] = 1

    # stratification reference distributions calculated on stratify_on
    p_ref = {}
    for c in stratify_on:
        p_ref[c] = class_prob(stratify_on[c])

    # best train and test indices in X; best associated score
    train_i, test_i, best_sco = None, None, np.inf

    # data size
    N = len(y)

    # full target coverage in all partitions
    full_target_coverage = False

    # categorical target: number of classes for coverage test
    if is_categorical(y[0]):
        nc = len(set(y))
    else:
        nc = None

    # brute-force optimization of SPLIT_ON split
    for tri, tei in gss.split(X, y, split_on):
        # all classes maintained in all partitions?
        if nc:
            nc_train = len(set(y[tri]))
            nc_test = len(set(y[tei]))
            if min(nc_train, nc_test) < nc:
                continue

        full_target_coverage = True

        sco = calc_split_score(tei, stratify_on, weight, p_ref, N, test_size)
        if sco < best_sco:
            train_i, test_i, best_sco = tri, tei, sco

    if test_i is None:
        sys.exit(exit_message(full_target_coverage))

    # matching info
    info = {
        "score": best_sco,
        "size_testset_in_spliton": test_size,
        "size_testset_in_X": np.round(len(test_i) / N, 2),
    }

    for c in p_ref:
        info[f"p_{c}_ref"] = p_ref[c]
        info[f"p_{c}_test"] = class_prob(stratify_on[c][test_i])

    return train_i, test_i, info


def calc_split_score(
    test_i, stratify_on, weight, p_ref, N, test_size, dev_i=None, dev_size=None
):
    """calculate split score based on class distribution IRADs and
    differences in partition sizes of groups vs observations; smaller is better.
    If dev_i and dev_size are not provided, the score is calculated for the train/test
    split only. If they are provided the score is calculated for the train/dev/test split
    Args:
    test_i: (np.array) of test set indices
    stratify_on: (dict) Dict-keys are variable names (targets and/or further groupings)
      the split should be stratified on (groupings could e.g. be sex, age class, etc).
      Dict-Values are np.array-s of length N that contain the variable values.
    weight: (dict) weight for each variable in stratify_on. Additional
      key: "size_diff" that weights the grouping vs observation level test set size difference
    p_ref: (dict) reference class distributions for all variables in stratify_on
    N: (int) size of underlying data set
    test_size: (float) test proportion in value set of variable, the disjunct grouping
       has been carried out
    dev_i: (np.array) of dev test indices
    dev_size: (float) dev proportion in value set of variable, the disjunct grouping
       has been carried out (this value should have been adjusted after splitting off the
       test set)
    """

    if dev_i is None:
        do_dev = False
    else:
        do_dev = True

    # dev and test set class distributions
    p_test, p_dev = {}, {}
    for c in p_ref:
        p_test[c] = class_prob(stratify_on[c][test_i])
        if do_dev:
            p_dev[c] = class_prob(stratify_on[c][dev_i])

    # score
    sco, wgt = 0, 0

    # IRADs (if p_test[c] or p_dec[c] do not contain
    # all classes in p_ref[c], return INF)
    for c in p_ref:
        irad, full_coverage = calc_irad(p_ref[c], p_test[c])
        if not full_coverage:
            return np.inf
        if do_dev:
            irad_dev, full_coverage = calc_irad(p_ref[c], p_dev[c])
            if not full_coverage:
                return np.inf
            irad = max(irad, irad_dev)

        sco += weight[c] * irad
        wgt += weight[c]

    # partition size difference groups vs observations
    size_diff = np.abs(len(test_i) / N - test_size)
    if do_dev:
        size_diff_dev = np.abs(len(dev_i) / N - dev_size)
        size_diff = max(size_diff, size_diff_dev)

    sco += weight["size_diff"] * size_diff
    wgt += weight["size_diff"]

    sco /= wgt

    return sco


def calc_irad(p1, p2):
    """calculate information radius of prob dicts p1 and p2
    Args:
    p1, p2: (dict) of probabilities
    Returns:
    ir: (float) information radius
    full_coverage: (bool) True if all elements in p1 occur in p2
        and vice versa
    """

    p, q = [], []
    full_coverage = True

    for u in sorted(p1.keys()):
        if u not in p2:
            full_coverage = False
            a = 0.0
        else:
            a = p2[u]

        p.append(p1[u])
        q.append(a)

    if full_coverage:
        if len(p2.keys()) > len(p1.keys()):
            full_coverage = False

    irad = ssp.distance.jensenshannon(p, q)

    return irad, full_coverage


def class_prob(y):
    """returns class probabilities in y
    Args:
    y (array-like) of classes
    Returns:
    p (dict) assigning to each class in Y its maximum likelihood
    """

    p = {}
    N = len(y)
    c = Counter(y)
    for x in c:
        p[x] = c[x] / N

    return p


def is_categorical(x):
    """returns True if type of x is in str or int*,
    else False"""

    if type(x) in [
        str,
        int,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
    ]:
        return True
    return False


def dummy_variable(X, columns, specs=None, squeeze_classes=False):
    """
    creates dummy variable from binned numeric columns that can be used
    later for stratification etc.

    Args:
    X: (pd.DataFrame)
    columns: (str or list) of numeric column names
    specs: (dict or str)
       if nested dict: keys are column names with subdict that contains the
           arguments for binning(), i.e. n_bins and lower_boundaries
    squeeze_classes: (boolean) further squeeze classes by sorting the digits
        within the string.
        Example: from binning of 3 columns, each into 2 bins, we got
                 "000", "100", "010", "001", "110", "101", "011", "111".
                 These classes are further squeezed by within-string sorting:
                 "000", "001", "011", "111"

    Returns:
    y: (list) of class strings of length X.shape[0]

    """

    df_bin = pd.DataFrame()
    if specs is None:
        specs = {}
    if type(columns) is str:
        columns = [columns]

    # bin columns
    for col in columns:
        if col not in X.columns:
            sys.exit(f"column {col} not in dataframe")
        if col in specs:
            kwargs = specs[col]
        else:
            kwargs = {"nbins": 2}
        yc = binning(X[col].to_numpy(), **kwargs)
        df_bin[col] = yc.astype(str)

    # concatenate
    df_bin["binvar"] = ""
    for col in columns:
        df_bin["binvar"] += df_bin[col]

    # squeeze
    if squeeze_classes:

        def squeezing(x):
            return "".join(sorted(x))

        df_bin["binvar"] = df_bin["binvar"].apply(squeezing)

    y = df_bin["binvar"].tolist()
    return y


def binning(y, nbins=3, lower_boundaries=None):
    """
    bins numeric array y either intrinsically into nbins classes
    based on an equidistant percentile split, or extrinsically
    by using the lower_boundaries values.

    Args:
    y: (np.array) with numeric data
    nbins: (int) number of bins
    lower_boundaries: (list) of lower bin boundaries.
      If provided nbins will be ignored and y is binned
      extrinsically. The first value of lower_boundaries
      is always corrected not to be higher than min(y).
    Returns:
    yc: (np.array) with bin IDs (integers from 0 to nbins-1)
    """

    # intrinsic binning by equidistant percentiles
    if lower_boundaries is None:
        prct = np.linspace(0, 100, nbins + 1)
        lower_boundaries = np.percentile(y, prct)
        lower_boundaries = lower_boundaries[0:nbins]
    else:
        # make sure that entire range of y is covered
        lower_boundaries[0] = min(lower_boundaries[0], np.min(y))

    # binned array
    yc = np.zeros(len(y), dtype=int)
    for i in range(1, len(lower_boundaries)):
        yc[y >= lower_boundaries[i]] = i

    return yc


def optimize_testset_split(
    X, y, split_on, stratify_on, weight=None, test_size=0.1, k=30, seed=42
):
    """backward compatibility"""
    return optimize_traintest_split(
        X, y, split_on, stratify_on, weight, test_size, k, seed
    )


def exit_message(full_target_coverage, infx="test"):
    if not full_target_coverage:
        return (
            "not all partitions contain all target classes. What you can do:\n"
            "(1) increase your dev and/or test partition, or\n"
            "(2) reduce the amount of target classes by merging some of them."
        )

    return (
        f"\n:-o No {infx} set split found. Reason is, that for at least one of the\n"
        f"stratification variables not all its values can make it into the {infx} set.\n"
        f"This happens e.g. if the {infx} set size is chosen too small or\n"
        "if the (multidimensional) distribution of the stratification\n"
        "variables is sparse. What you can do:\n"
        "(1) remove a variable from this stratification, or\n"
        "(2) merge classes within a variable to increase the per class probabilities, or\n"
        f"(3) increase the {infx} set size, or\n"
        "(4) increase the number of different splits (if it was small, say < 10, before), or\n"
        "(5) in case your target is numeric and you have added a binned target array to the\n"
        "    stratification variables: reduce the number of bins.\n"
        "Good luck!\n"
    )
