"""Percentogram: an equal-mass (quantile-width) histogram.

A normal histogram uses equal-*width* bins, so a single outlier can leave most
bins nearly empty and hide the shape of the bulk of the distribution. A
percentogram instead uses equal-*mass* bins (each bin holds the same fraction
of observations) and plots density (height = 1 / bin width) rather than raw
counts, so the area of every bar is the same and the shape of the distribution
stays readable even with heavy skew or extreme outliers.

No existing package (searched: matplotlib, seaborn, statsmodels, plotnine)
was found to ship this as a named, reusable function, so it's kept here
rather than pointed at a dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes


def percentogram(
    series: pd.Series,
    num_bins: int = 10,
    ax: Axes | None = None,
    **bar_kwargs,
) -> Axes:
    """Plot an equal-mass (quantile-width) histogram of ``series``.

    Args:
        series: Numeric values to plot. NaNs are dropped.
        num_bins: Target number of equal-mass bins. Fewer bins are used if
            the data doesn't have enough distinct quantile edges (e.g. a
            heavily duplicated / low-cardinality series).
        ax: Existing matplotlib Axes to draw on. A new figure and axes are
            created if not given.
        **bar_kwargs: Extra keyword arguments forwarded to ``ax.bar`` (e.g.
            ``color=...``, ``edgecolor=...``).

    Returns:
        The matplotlib Axes the percentogram was drawn on.

    Raises:
        ValueError: If ``series`` has fewer than 2 non-NaN values, or
            ``num_bins`` is not a positive integer.
    """
    if num_bins < 1:
        raise ValueError(f"num_bins must be a positive integer, got {num_bins}")

    clean = pd.Series(series).dropna()
    if len(clean) < 2:
        raise ValueError(
            f"percentogram needs at least 2 non-NaN observations, got {len(clean)}"
        )

    quantiles = np.linspace(0, 1, num_bins + 1)
    bins = np.unique(np.quantile(clean, quantiles))
    if len(bins) < 2:
        # All values identical - draw a single full-mass bar centered on the value.
        bins = np.array([clean.iloc[0] - 0.5, clean.iloc[0] + 0.5])

    widths = bins[1:] - bins[:-1]
    counts, _ = np.histogram(clean, bins=bins)
    heights = (counts / len(clean)) / widths
    centers = (bins[:-1] + bins[1:]) / 2

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

    bar_kwargs.setdefault("align", "center")
    ax.bar(centers, heights, width=widths, **bar_kwargs)
    ax.set_title("Percentogram")
    ax.set_xlabel(series.name if series.name else "Value")
    ax.set_ylabel("Density (equal-mass bins)")
    return ax
