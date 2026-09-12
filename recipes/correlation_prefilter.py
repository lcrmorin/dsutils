"""Drop near-duplicate features before feature_interactions.run_screen -
already a scikit-learn transformer in feature_engine, no need to build one.

Checked before writing anything: `feature_engine.selection` already ships
two correlation-based selectors, both real `BaseEstimator` +
`TransformerMixin` subclasses (verified: `isinstance(transformer,
(BaseEstimator, TransformerMixin))` is True for both, and both have a
working `fit`/`fit_transform`):

- `DropCorrelatedFeatures(threshold=0.8)` - for each cluster of features
  correlated above `threshold`, keeps the first one encountered, drops the
  rest. Simple, arbitrary tie-break.
- `SmartCorrelatedSelection(threshold=0.8, selection_method=...)` - same
  clustering, but picks which feature to KEEP per cluster by a criterion
  (`"variance"`, `"cardinality"`, `"missing_values"`, or `"model_performance"`
  with an estimator + scoring metric) instead of "whichever came first."
  Exposes `.correlated_feature_sets_` - the actual clusters found, useful
  to log/report on its own.

Verified on a synthetic case (`a`, `b` correlated at ~0.9999, `c`
independent): both correctly identify `{a, b}` as one cluster and drop
exactly one of them, keeping `c`. `SmartCorrelatedSelection` is the better
fit for the `feature_interactions` prefilter use case specifically, since
"keep whichever has more variance / predicts the target better" is a more
defensible choice than "keep whichever was listed first" when you're about
to spend O(k^2) time on the survivors.
"""

from feature_engine.selection import SmartCorrelatedSelection


def correlation_prefilter(threshold=0.8, selection_method="variance", estimator=None, scoring="roc_auc"):
    """Build a fitted-on-call transformer that drops one of each pair of
    features correlated above `threshold`, keeping the "best" one per
    cluster by `selection_method`.

    Usage:
        prefilter = correlation_prefilter(threshold=0.9)
        df_reduced = prefilter.fit_transform(df[numeric_cols])
        kept_cols = list(df_reduced.columns)
        result = run_screen(df, targets, kept_cols)   # O(k^2) now runs on fewer k
        print(prefilter.correlated_feature_sets_)      # what got merged, for the record
    """
    kwargs = dict(threshold=threshold, selection_method=selection_method)
    if selection_method == "model_performance":
        if estimator is None:
            raise ValueError("selection_method='model_performance' requires an estimator=...")
        kwargs.update(estimator=estimator, scoring=scoring)
    return SmartCorrelatedSelection(**kwargs)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(size=n)
    b = a + rng.normal(scale=0.01, size=n)  # near-duplicate of a
    c = rng.normal(size=n)                  # independent
    df = pd.DataFrame({"a": a, "b": b, "c": c})

    prefilter = correlation_prefilter(threshold=0.9)
    reduced = prefilter.fit_transform(df)
    print("kept:", list(reduced.columns))
    print("dropped:", prefilter.features_to_drop_)
    print("correlated clusters found:", prefilter.correlated_feature_sets_)
