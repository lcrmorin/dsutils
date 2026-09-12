import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from recipes.correlation_prefilter import correlation_prefilter


def test_is_a_real_sklearn_transformer():
    prefilter = correlation_prefilter(threshold=0.9)
    assert isinstance(prefilter, BaseEstimator)
    assert isinstance(prefilter, TransformerMixin)


def test_drops_near_duplicate_and_keeps_independent_feature():
    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.01, size=n)  # correlation ~0.9999 with a
    c = rng.normal(size=n)                  # independent
    df = pd.DataFrame({"a": a, "b": b, "c": c})

    prefilter = correlation_prefilter(threshold=0.9)
    reduced = prefilter.fit_transform(df)

    assert "c" in reduced.columns
    assert len({"a", "b"} & set(reduced.columns)) == 1  # exactly one of a/b survives
    assert prefilter.correlated_feature_sets_ == [{"a", "b"}]


def test_no_drop_when_nothing_correlated_above_threshold():
    rng = np.random.default_rng(1)
    n = 1000
    df = pd.DataFrame({c: rng.normal(size=n) for c in ["x", "y", "z"]})

    prefilter = correlation_prefilter(threshold=0.9)
    reduced = prefilter.fit_transform(df)

    assert set(reduced.columns) == {"x", "y", "z"}
    assert prefilter.correlated_feature_sets_ == []
