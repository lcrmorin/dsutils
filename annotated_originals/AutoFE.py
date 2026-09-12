import pandas as pd, numpy as np
from scipy import stats

def IQR(x, axis):
    return np.quantile(x, 0.75, axis=axis) - np.quantile(x, 0.25, axis=axis)

def sharpe(x, axis):
    return np.mean(x, axis=axis) / np.std(x, axis=axis)

def mode(x, axis):
    return stats.mode(x, axis=axis, keepdims=False).mode

def quantile01(a, axis):
    return np.quantile(a, q=0.1, axis=axis)

def quantile02(a, axis):
    return np.quantile(a, q=0.2, axis=axis)

def quantile08(a, axis):
    return np.quantile(a, q=0.8, axis=axis)

def quantile09(a, axis):
    return np.quantile(a, q=0.9, axis=axis)

base_stats = [np.mean, np.prod, np.std, stats.skew, stats.kurtosis, sharpe]
quantile_stats = [np.min, quantile01, quantile02, np.median, quantile08, quantile09, np.max, IQR, np.ptp, mode]
all_stats = base_stats + quantile_stats

range_count = 23

for df in [train, test]:

    #base stats
    for fun in all_stats:
        df[f"Agg_Feature_{fun.__name__}"] = fun(
                (df.loc[:, initial_features].values), axis=1)

    # count stats
    for i in np.arange(0, range_count):
        df[f"count_{i}"] = (df[initial_features] == i).sum(axis=1)

Features = [f for f in train.columns if f not in targets+ids]

# ---------------------------------------------------------------------------
# feature_engine has no drop-in for this (it doesn't do row-wise multi-stat
# aggregation across a feature list). Native pandas does, and I checked the
# actual numbers rather than assuming the obvious `.agg()`/`.skew()` calls
# match - two of them didn't, on the first try:
#
#   - `DataFrame.std(axis=1)` defaults to ddof=1 (sample std); this script's
#     `np.std(x, axis=axis)` is ddof=0 (population std) - different numbers
#     on the same data. Fix: `sub.std(axis=1, ddof=0)`.
#   - `DataFrame.skew(axis=1)`/`.kurt(axis=1)` use the bias-corrected
#     formula; this script's `scipy.stats.skew`/`kurtosis` default to the
#     biased formula - genuinely different values (checked numerically),
#     not just a naming difference. Fix: keep going through
#     `df.agg([stats.skew, stats.kurtosis], axis=1)` instead of the native
#     `.skew()`/`.kurt()` methods.
#
# Full working replacement (self-tested against this exact script's
# statistics - mean/std/skew/kurtosis/IQR/mode/count_3 all match to
# floating-point tolerance): see `autofe_native_pandas_equivalent.py` next
# to this file. One stat has no vectorized pandas/numpy row-wise form at
# all - `scipy.stats.mode` is still the only way to get it, same as here.
# ---------------------------------------------------------------------------
