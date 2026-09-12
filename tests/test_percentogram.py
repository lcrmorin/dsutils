import matplotlib

matplotlib.use("Agg")  # no display in CI/sandbox

import numpy as np
import pandas as pd
import pytest

from dsutils.dataviz.percentogram import percentogram


def test_basic_shape_matches_bin_count():
    s = pd.Series(np.random.default_rng(0).normal(size=1000))
    ax = percentogram(s, num_bins=10)
    # one bar per bin edge-to-edge
    assert len(ax.patches) == 10


def test_equal_mass_bins_have_roughly_equal_area():
    s = pd.Series(np.random.default_rng(1).exponential(size=2000))
    ax = percentogram(s, num_bins=8)
    areas = [p.get_height() * p.get_width() for p in ax.patches]
    # equal-mass bins -> each bar's area (~fraction of data) should be close to 1/8
    assert max(areas) / min(areas) < 1.5


def test_outlier_does_not_collapse_bulk_of_bins():
    # a normal histogram with equal-width bins would put almost everything in
    # one bin here; a percentogram should not.
    s = pd.Series(np.concatenate([np.random.default_rng(2).normal(size=999), [10_000]]))
    ax = percentogram(s, num_bins=10)
    widths = [p.get_width() for p in ax.patches]
    assert max(widths) / min(widths) > 100  # the outlier bin is much wider, not the others


def test_nan_values_are_dropped():
    s = pd.Series([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
    ax = percentogram(s, num_bins=2)
    assert len(ax.patches) >= 1


def test_constant_series_draws_single_bar_without_crashing():
    s = pd.Series([5.0] * 20)
    ax = percentogram(s, num_bins=5)
    assert len(ax.patches) == 1


def test_too_few_observations_raises():
    with pytest.raises(ValueError):
        percentogram(pd.Series([1.0]), num_bins=5)


def test_invalid_num_bins_raises():
    with pytest.raises(ValueError):
        percentogram(pd.Series([1.0, 2.0, 3.0]), num_bins=0)


def test_num_bins_is_actually_respected():
    # regression test: the original script silently ignored num_bins and
    # always used q=10.
    s = pd.Series(np.random.default_rng(3).normal(size=5000))
    ax = percentogram(s, num_bins=4)
    assert len(ax.patches) == 4
