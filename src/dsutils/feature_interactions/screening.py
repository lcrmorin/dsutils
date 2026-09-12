"""Generic pairwise feature-interaction screening.

For every pair of numeric features and a small set of combining operators,
quantile-bin the combined value and measure how much of a target's variance
that binning explains (eta-squared / correlation ratio), compared to what
each feature explains alone. A large "gap" (combo eta2 well above the sum of
the two solo eta2's) flags a pair + operator worth engineering as an actual
feature - the combination captures target variance neither input does on
its own.

Built from a user-supplied `run_screen()` skeleton; the pieces below
(`OPS`, `combine`, `rank_quantile`, `eta2_from_ranks`, `base_eta2`) fill in
what it calls. Quantile binning follows the same convention already used
elsewhere in this project (IV_quantile.py, plot_feature_binning.py):
`np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))`, so behavior on
duplicate/degenerate quantiles is consistent across all three.

Eta-squared here is the standard one-way ANOVA correlation ratio,
`SS_between / SS_total` for `y` grouped by bin membership - verified against
`statsmodels.formula.api.ols(...).fit()` / `sm.stats.anova_lm` on synthetic
grouped data in tests/test_feature_interactions.py (exact match to floating
point tolerance).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

OPS = ["add", "sub", "mul", "div"]


def combine(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    """Elementwise combination of two numeric arrays. `div` produces inf/nan
    on zero denominators by design - callers are expected to mask those out
    with `np.isfinite`, same as `run_screen` already does."""
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "div":
        with np.errstate(divide="ignore", invalid="ignore"):
            return a / b
    raise ValueError(f"Unknown op: {op!r}. Expected one of {OPS}.")


def rank_quantile(x: np.ndarray, n_bins: int) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Bin `x` (already finite, 1D) into up to `n_bins` equal-frequency
    quantile bins. Returns (bin_index_per_row, n_bins_actual), or (None,
    None) if there aren't enough distinct quantile edges to form at least 2
    bins (mirrors IV_quantile.py's `len(quantiles) <= 2: return 0` guard)."""
    quantiles = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) <= 2:
        return None, None
    edges = quantiles[1:-1]
    ranks = np.digitize(x, edges)
    n_bins_actual = len(quantiles) - 1
    return ranks, n_bins_actual


def eta2_from_ranks(ranks: np.ndarray, n_bins_actual: int, y: np.ndarray) -> float:
    """Correlation ratio (eta-squared) of `y` explained by group membership
    `ranks` (integers in [0, n_bins_actual)). 0 = groups carry no
    information about y's level; 1 = y is constant within every group."""
    grand_mean = y.mean()
    ss_total = np.sum((y - grand_mean) ** 2)
    if ss_total == 0:
        return 0.0
    group_counts = np.bincount(ranks, minlength=n_bins_actual)
    group_sums = np.bincount(ranks, weights=y, minlength=n_bins_actual)
    nonempty = group_counts > 0
    group_means = np.zeros(n_bins_actual)
    group_means[nonempty] = group_sums[nonempty] / group_counts[nonempty]
    ss_between = np.sum(group_counts[nonempty] * (group_means[nonempty] - grand_mean) ** 2)
    return float(ss_between / ss_total)


def base_eta2(df: pd.DataFrame, targets, numeric_cols, n_bins: int = 100) -> dict:
    """eta2_single[target_name][col] = eta2 of `col` alone (quantile-binned)
    against that target, restricted to rows finite in both."""
    result: dict = {}
    for name, y in targets:
        y = np.asarray(y)
        result[name] = {}
        for col in numeric_cols:
            x = df[col].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 30:
                result[name][col] = np.nan
                continue
            ranks, n_bins_actual = rank_quantile(x[mask], n_bins)
            if ranks is None:
                result[name][col] = np.nan
                continue
            result[name][col] = eta2_from_ranks(ranks, n_bins_actual, y[mask])
    return result


def run_screen(df, targets, numeric_cols, n_bins=100, log_every_ops=True):
    """Loop: target x horizon (folded into `targets`) x every feature pair x operator. One
    quantile-binning pass per pair x op, shared across all targets. Returns one long DataFrame:
    var_a, var_b, op, target, combo_eta2, eta2_a, eta2_b, gap."""
    numeric_cols = list(numeric_cols)
    df = df.reset_index(drop=True)
    test_vals = {c: df[c].to_numpy() for c in numeric_cols}

    eta2_single = base_eta2(df, targets, numeric_cols, n_bins)

    rows = []
    t0 = time.time()
    n_pairs = len(numeric_cols) * (len(numeric_cols) - 1) // 2
    for op in OPS:
        n_done = 0
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                combo = combine(test_vals[a], test_vals[b], op)
                mask = np.isfinite(combo)
                n_done += 1
                if mask.sum() < 30:
                    continue
                ranks, n_bins_actual = rank_quantile(combo[mask], n_bins)
                if ranks is None:
                    continue
                for name, y in targets:
                    y_arr = np.asarray(y)
                    combo_eta2 = eta2_from_ranks(ranks, n_bins_actual, y_arr[mask])
                    ea, eb = eta2_single[name][a], eta2_single[name][b]
                    rows.append({
                        "var_a": a, "var_b": b, "op": op, "target": name,
                        "combo_eta2": combo_eta2, "eta2_a": ea, "eta2_b": eb,
                        "gap": combo_eta2 - ea - eb,
                    })
        if log_every_ops:
            print(f"  op {op}: {n_done}/{n_pairs} pairs done, elapsed={time.time() - t0:.0f}s")
    return pd.DataFrame(rows)
