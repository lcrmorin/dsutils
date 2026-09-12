"""Native-pandas equivalent of the original AutoFE.py - no feature_engine needed.

Two corrections versus my first pass, both caught by actually checking the
numbers rather than assuming "native method with the same name = same result":

1. `DataFrame.std(axis=1)` defaults to `ddof=1` (sample std); the original
   script used `np.std(x, axis=axis)`, which is `ddof=0` (population std).
   Different numbers for the same data - pass `ddof=0` to match exactly.
2. `DataFrame.skew(axis=1)` / `.kurt(axis=1)` use the bias-*corrected* sample
   formula. The original script's `scipy.stats.skew`/`kurtosis` default to
   the *biased* (uncorrected) formula. These genuinely disagree (checked
   numerically, not close for small row-widths) - so exact equivalence still
   goes through `scipy.stats.skew`/`kurtosis`, just via `df.agg([...], axis=1)`
   instead of a manual per-row loop. Pandas' own `.skew()`/`.kurt()` are
   arguably the more standard estimator if you don't need to match the
   original exactly - use whichever you actually want.

The only stat with no vectorized pandas/numpy equivalent at all is row-wise
mode, which still goes through `scipy.stats.mode`.
"""

import numpy as np
import pandas as pd
from scipy import stats


def add_autofe_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    count_range: range = range(23),
) -> None:
    """Add the same aggregate + per-value-count features as AutoFE.py, in place."""
    sub = df[feature_cols]
    vals = sub.values

    # --- native DataFrame reducers: no scipy, no custom function needed ---
    df["Agg_Feature_mean"] = sub.mean(axis=1)
    df["Agg_Feature_prod"] = sub.prod(axis=1)
    df["Agg_Feature_std"] = sub.std(axis=1, ddof=0)  # ddof=0 to match np.std exactly
    skew_kurt = sub.agg([stats.skew, stats.kurtosis], axis=1)  # exact match; see module docstring
    df["Agg_Feature_skew"] = skew_kurt["skew"]
    df["Agg_Feature_kurtosis"] = skew_kurt["kurtosis"]
    df["Agg_Feature_sharpe"] = df["Agg_Feature_mean"] / df["Agg_Feature_std"]  # was a custom function

    # --- quantile-based stats: one vectorized np.quantile call for all of them ---
    q_levels = [0.0, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 1.0]
    q_names = ["min", "quantile01", "quantile02", "_q25", "median", "_q75", "quantile08", "quantile09", "max"]
    q = dict(zip(q_names, np.quantile(vals, q_levels, axis=1)))
    for name, row in q.items():
        if not name.startswith("_"):
            df[f"Agg_Feature_{name}"] = row
    df["Agg_Feature_IQR"] = q["_q75"] - q["_q25"]      # was a custom IQR() function
    df["Agg_Feature_ptp"] = q["max"] - q["min"]         # was np.ptp

    # --- mode: the one stat with no vectorized pandas/numpy row-wise equivalent ---
    df["Agg_Feature_mode"] = stats.mode(vals, axis=1, keepdims=False).mode

    # --- per-value counts: already a one-liner, unchanged ---
    for i in count_range:
        df[f"count_{i}"] = (sub == i).sum(axis=1)


if __name__ == "__main__":
    # quick self-check against the exact statistics the original script computed
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.integers(0, 10, size=(50, 6)), columns=[f"f{i}" for i in range(6)])
    feature_cols = list(df.columns)
    add_autofe_features(df, feature_cols, count_range=range(0, 10))

    vals = df[feature_cols].values
    checks = {
        "mean": (df["Agg_Feature_mean"].values, np.mean(vals, axis=1)),
        "std": (df["Agg_Feature_std"].values, np.std(vals, axis=1)),
        "skew": (df["Agg_Feature_skew"].values, stats.skew(vals, axis=1)),
        "kurtosis": (df["Agg_Feature_kurtosis"].values, stats.kurtosis(vals, axis=1)),
        "IQR": (df["Agg_Feature_IQR"].values,
                 np.quantile(vals, 0.75, axis=1) - np.quantile(vals, 0.25, axis=1)),
        "mode": (df["Agg_Feature_mode"].values, stats.mode(vals, axis=1, keepdims=False).mode),
        "count_3": (df["count_3"].values, (vals == 3).sum(axis=1)),
    }
    for name, (got, want) in checks.items():
        assert np.allclose(got, want, equal_nan=True), f"{name} mismatch"
    print(f"All {len(checks)} checks match the original script's statistics.")
