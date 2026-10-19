import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from dsutils.time_series import scan_trends


def test_each_window_matches_independent_statsmodels_ols():
    rng = np.random.default_rng(42)
    y = .25 * np.arange(60) + rng.normal(size=60)
    result = scan_trends(y, min_lookback=5, max_lookback=40, step=7)
    assert result.index.tolist() == [5, 12, 19, 26, 33, 40]
    for length, row in result.iterrows():
        reference = sm.OLS(y[-length:], sm.add_constant(np.arange(length))).fit()
        np.testing.assert_allclose([row.intercept, row.slope], reference.params, rtol=1e-10)
        np.testing.assert_allclose(row.slope_stderr, reference.bse[1], rtol=1e-10)
        np.testing.assert_allclose(row.t_stat, reference.tvalues[1], rtol=1e-10)
        np.testing.assert_allclose(row.r_squared, reference.rsquared, rtol=1e-10)


def test_direction_reversal_changes_sign_not_strength():
    y = np.arange(30) + np.random.default_rng(7).normal(size=30)
    up, down = scan_trends(y), scan_trends(-y)
    np.testing.assert_allclose(up.t_stat, -down.t_stat)
    assert up.t_stat.abs().idxmax() == down.t_stat.abs().idxmax()
    assert down.slope.lt(0).all()


def test_series_labels_and_trailing_window_without_mutation():
    values = pd.Series([1000., -1000., 2., 4., 6., 8.], index=pd.date_range("2020-01-01", periods=6))
    original = values.copy()
    row = scan_trends(values, min_lookback=4, max_lookback=4).iloc[0]
    assert row.start == values.index[-4] and row.end == values.index[-1]
    assert row.slope == pytest.approx(2.) and row.intercept == pytest.approx(2.)
    pd.testing.assert_series_equal(values, original)


def test_constant_and_exact_linear_conventions():
    constant = scan_trends(np.ones(12))
    assert (constant[["slope", "slope_stderr", "t_stat", "r_squared"]] == 0).all().all()
    row = scan_trends(-2 * np.arange(12), min_lookback=12).iloc[0]
    assert row.slope == -2 and row.slope_stderr == 0
    assert row.t_stat == -np.inf and row.r_squared == 1


@pytest.mark.parametrize("values", [[], [1, 2], [[1, 2, 3]], [1, np.nan, 3], [1, np.inf, 3]])
def test_invalid_observations(values):
    with pytest.raises(ValueError, match="values must"):
        scan_trends(values)


@pytest.mark.parametrize("kwargs", [
    {"min_lookback": 2}, {"max_lookback": 11}, {"min_lookback": 8, "max_lookback": 5},
    {"step": 0}, {"step": -1}, {"min_lookback": 3.5}, {"max_lookback": True}, {"step": False},
])
def test_invalid_window_parameters(kwargs):
    with pytest.raises(ValueError):
        scan_trends(np.arange(10), **kwargs)
