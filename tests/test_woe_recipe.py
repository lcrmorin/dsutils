import numpy as np
import pandas as pd
import pytest

from recipes.woe_encoding import woe_encoding_pipeline


def _iv_quantile_bins(x, bins=10):
    quantiles = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    return np.digitize(x, quantiles[1:-1])


def test_binning_matches_iv_quantile_exactly():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x})

    expected_bins = _iv_quantile_bins(x, bins=10)

    pipe = woe_encoding_pipeline(["x"], n_bins=10)
    got_bins = pipe.named_steps["bin"].fit_transform(df)["x"].to_numpy()

    assert np.array_equal(got_bins, expected_bins)


def test_woe_values_match_manual_formula_when_unregularized():
    rng = np.random.default_rng(1)  # picked to avoid a pure (single-class) bin
    n = 5000
    x = rng.normal(size=n)
    y = pd.Series((x + rng.normal(scale=0.7, size=n) > 0).astype(int))
    df = pd.DataFrame({"x": x})

    bins = _iv_quantile_bins(x, bins=10)
    manual_woe = {}
    for b in np.unique(bins):
        mask = bins == b
        pos = y[mask].sum()
        neg = (1 - y[mask]).sum()
        assert pos > 0 and neg > 0  # sanity: no pure bin in this run
        manual_woe[b] = np.log((pos / y.sum()) / (neg / (1 - y).sum()))
    manual_series = pd.Series(bins).map(manual_woe)

    pipe = woe_encoding_pipeline(["x"], n_bins=10, regularization=0.0)
    woe = pipe.fit_transform(df, y)["x"]

    assert np.allclose(manual_series.to_numpy(), woe.to_numpy(), rtol=1e-9)


def test_default_regularization_avoids_infinite_woe_on_pure_bin():
    # construct data where the top bin is deliberately pure (all y=1)
    rng = np.random.default_rng(2)
    n = 500
    x = np.concatenate([rng.normal(size=n - 50), rng.normal(loc=10, size=50)])
    y = pd.Series(np.concatenate([np.zeros(n - 50), np.ones(50)]).astype(int))
    df = pd.DataFrame({"x": x})

    pipe_reg = woe_encoding_pipeline(["x"], n_bins=10)  # default regularization=1.0
    woe_reg = pipe_reg.fit_transform(df, y)["x"]
    assert np.isfinite(woe_reg).all()

    pipe_noreg = woe_encoding_pipeline(["x"], n_bins=10, regularization=0.0)
    woe_noreg = pipe_noreg.fit_transform(df, y)["x"]
    assert not np.isfinite(woe_noreg).all()  # confirms the pure bin actually triggers inf without smoothing
