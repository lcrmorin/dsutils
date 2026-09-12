import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from dsutils.feature_interactions import (
    OPS,
    base_eta2,
    combine,
    eta2_from_ranks,
    rank_quantile,
    run_screen,
)


def test_combine_matches_elementwise_numpy():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 0.0, 1.0, -2.0])
    assert np.allclose(combine(a, b, "add"), a + b)
    assert np.allclose(combine(a, b, "sub"), a - b)
    assert np.allclose(combine(a, b, "mul"), a * b)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_div = a / b
    got_div = combine(a, b, "div")
    assert np.allclose(got_div, expected_div, equal_nan=True)
    assert not np.isfinite(got_div[1])  # 2.0 / 0.0


def test_combine_rejects_unknown_op():
    with pytest.raises(ValueError):
        combine(np.array([1.0]), np.array([1.0]), "xor")


def test_rank_quantile_matches_iv_quantile_convention():
    # same construction as IV_quantile.py / plot_feature_binning.py
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    ranks, n_bins_actual = rank_quantile(x, n_bins=10)
    quantiles = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
    expected_ranks = np.digitize(x, quantiles[1:-1])
    assert n_bins_actual == len(quantiles) - 1
    assert np.array_equal(ranks, expected_ranks)


def test_rank_quantile_degenerate_returns_none():
    # a near-constant array collapses to <=2 unique quantile edges
    x = np.zeros(500)
    ranks, n_bins_actual = rank_quantile(x, n_bins=10)
    assert ranks is None and n_bins_actual is None


def test_eta2_matches_statsmodels_one_way_anova():
    """eta2_from_ranks should equal the standard ANOVA eta-squared
    (SS_between / SS_total) computed independently via statsmodels' OLS +
    anova_lm, on the same grouped data."""
    rng = np.random.default_rng(1)
    n_groups = 5
    n_per_group = 200
    group_means = np.array([0.0, 1.0, 2.0, 1.5, -1.0])
    ranks = np.repeat(np.arange(n_groups), n_per_group)
    y = group_means[ranks] + rng.normal(scale=2.0, size=n_groups * n_per_group)

    got = eta2_from_ranks(ranks, n_groups, y)

    df = pd.DataFrame({"y": y, "grp": pd.Categorical(ranks)})
    model = smf.ols("y ~ C(grp)", data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    ss_between = anova.loc["C(grp)", "sum_sq"]
    ss_total = ss_between + anova.loc["Residual", "sum_sq"]
    expected = ss_between / ss_total

    assert got == pytest.approx(expected, rel=1e-9)


def test_eta2_zero_when_groups_carry_no_information():
    rng = np.random.default_rng(2)
    y = rng.normal(size=1000)
    ranks = rng.integers(0, 4, size=1000)  # random group assignment, independent of y
    got = eta2_from_ranks(ranks, 4, y)
    assert 0.0 <= got < 0.02  # should be near zero, allow sampling noise


def test_eta2_one_when_y_constant_within_group():
    ranks = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, -3.0, -3.0, -3.0])
    got = eta2_from_ranks(ranks, 3, y)
    assert got == pytest.approx(1.0)


def test_base_eta2_matches_manual_single_column_computation():
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.normal(size=n)
    y = x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x": x})

    result = base_eta2(df, [("y", y)], ["x"], n_bins=10)

    ranks, n_bins_actual = rank_quantile(x, 10)
    expected = eta2_from_ranks(ranks, n_bins_actual, y)
    assert result["y"]["x"] == pytest.approx(expected)


def test_run_screen_flags_a_real_multiplicative_interaction():
    """Classic interaction case: y depends on a*b, but a and b alone carry
    much less signal individually. The mul-combo's eta2 should clearly beat
    the sum of the two solo eta2's (a large positive `gap`), and it should
    be the top-ranked row for this target - the actual thing this tool is
    for."""
    rng = np.random.default_rng(4)
    n = 5000
    a = rng.uniform(-1, 1, size=n)
    b = rng.uniform(-1, 1, size=n)
    c = rng.normal(size=n)  # unrelated noise feature
    y = a * b + rng.normal(scale=0.05, size=n)

    df = pd.DataFrame({"a": a, "b": b, "c": c})
    result = run_screen(df, [("y", y)], ["a", "b", "c"], n_bins=20, log_every_ops=False)

    mul_ab = result[(result.var_a == "a") & (result.var_b == "b") & (result.op == "mul")]
    assert len(mul_ab) == 1
    row = mul_ab.iloc[0]

    assert row.combo_eta2 > 0.5          # a*b strongly explains y
    assert row.eta2_a < 0.05             # a alone barely does
    assert row.eta2_b < 0.05             # b alone barely does
    assert row.gap > 0.4                 # the interaction is the whole story

    top_row = result.sort_values("combo_eta2", ascending=False).iloc[0]
    assert {top_row.var_a, top_row.var_b} == {"a", "b"}
    assert top_row.op == "mul"


def test_run_screen_output_shape_and_columns():
    rng = np.random.default_rng(5)
    n = 500
    df = pd.DataFrame({c: rng.normal(size=n) for c in ["f1", "f2", "f3"]})
    y1 = df["f1"].to_numpy() + rng.normal(scale=1.0, size=n)
    y2 = df["f2"].to_numpy() + rng.normal(scale=1.0, size=n)

    result = run_screen(df, [("y1", y1), ("y2", y2)], ["f1", "f2", "f3"], n_bins=10, log_every_ops=False)

    expected_cols = {"var_a", "var_b", "op", "target", "combo_eta2", "eta2_a", "eta2_b", "gap"}
    assert expected_cols == set(result.columns)
    n_pairs = 3  # C(3,2)
    assert len(result) == n_pairs * len(OPS) * 2  # 2 targets
