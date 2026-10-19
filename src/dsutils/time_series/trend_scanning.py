"""Scan trailing linear trends at the last observation of a series."""

from numbers import Integral
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress


def scan_trends(
    values,
    min_lookback: int = 3,
    max_lookback: Optional[int] = None,
    step: int = 1,
) -> pd.DataFrame:
    """Fit each candidate trailing window with SciPy's ordinary least squares.

    Parameters
    ----------
    values : one-dimensional array-like or pandas.Series
        Finite observations in chronological order. Regressions use observation
        positions, not elapsed time; irregular timestamps are treated as equally
        spaced. A Series index is retained as start/end labels, not sorted.
    min_lookback, max_lookback : int
        Inclusive window bounds, with 3 <= min <= max <= len(values). The default
        maximum is the full series. Every window ends at the last observation.
    step : int
        Positive stride between candidate lengths. The maximum is included only
        if it lies on that stride.

    Returns
    -------
    pandas.DataFrame
        Indexed by increasing ``lookback``, with ``start``, ``end``, ``intercept``,
        ``slope``, ``slope_stderr``, ``t_stat`` and ``r_squared``. The intercept is
        at the first observation of each window (local x=0). Slopes are per
        observation. Constant windows use t_stat=0 and r_squared=0 by convention;
        a nonconstant fit with zero standard error has signed infinite t_stat.

    Notes
    -----
    Select the strongest direction with ``result.t_stat.abs().idxmax()``;
    ``result.t_stat.idxmax()`` reproduces the original signed ranking. Ties pick
    the shorter window. The signed ranking does not guarantee a positive slope.
    These are descriptive, unadjusted OLS statistics: searching windows inflates
    selection effects, and serially correlated residuals invalidate ordinary
    standard-error inference. No post-selection significance claim is made.
    Missing/infinite values are rejected rather than changing the time spacing.
    Nothing is plotted or fitted at import time.
    """
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all():
        raise ValueError("values must be one-dimensional, finite, and have at least 3 observations")
    if max_lookback is None:
        max_lookback = len(y)
    for name, value in (("min_lookback", min_lookback),
                        ("max_lookback", max_lookback), ("step", step)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if not 3 <= min_lookback <= max_lookback <= len(y):
        raise ValueError("require 3 <= min_lookback <= max_lookback <= len(values)")
    if step < 1:
        raise ValueError("step must be positive")

    labels = values.index if isinstance(values, pd.Series) else np.arange(len(y))
    rows = []
    for lookback in range(min_lookback, max_lookback + 1, step):
        window = y[-lookback:]
        if np.all(window == window[0]):
            intercept, slope, stderr, t_stat, r_squared = window[0], 0., 0., 0., 0.
        else:
            fit = linregress(np.arange(lookback, dtype=float), window)
            intercept, slope, stderr = fit.intercept, fit.slope, fit.stderr
            t_stat = slope / stderr if stderr > 0 else np.copysign(np.inf, slope)
            r_squared = fit.rvalue ** 2
        rows.append({"lookback": lookback, "start": labels[-lookback], "end": labels[-1],
                     "intercept": intercept, "slope": slope, "slope_stderr": stderr,
                     "t_stat": t_stat, "r_squared": r_squared})
    return pd.DataFrame(rows).set_index("lookback")
