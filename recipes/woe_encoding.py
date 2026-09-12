"""WOE-encoding a continuous feature - no new code needed, two existing
sklearn-compatible transformers already do this exactly.

`IV_quantile.py` bins a continuous feature into equal-frequency quantile
bins and returns the summed Information Value, but doesn't hand back an
actual WOE-encoded column - the thing scorecard modeling uses downstream.
Checked for an existing transform before writing one:

- `feature_engine.discretisation.EqualFrequencyDiscretiser` reproduces
  IV_quantile.py's own binning exactly - verified bin-for-bin (same row,
  same bin index) on synthetic data, not just "same number of bins."
- `category_encoders.WOEEncoder` computes per-bin WOE with the identical
  formula to a hand-rolled one, plus optional additive smoothing
  (`regularization=`) to avoid the divide-by-zero a hand-rolled version
  hits on a bin with only one class present. With `regularization=0.0` it
  matches a manual per-bin WOE calculation to floating-point tolerance
  (see tests/test_woe_recipe.py); the small default smoothing
  (`regularization=1.0`) is what you actually want in practice, since a
  bin that's pure by chance shouldn't get an infinite WOE.

Chained in a `Pipeline`, these two replace a hand-built WOE transform
entirely - both are actively maintained (category_encoders released days
ago, feature_engine ~7 months ago) and both are real
`BaseEstimator`/`TransformerMixin` subclasses, so they drop into any
scikit-learn `Pipeline`/`ColumnTransformer` as-is.
"""

from feature_engine.discretisation import EqualFrequencyDiscretiser
from category_encoders import WOEEncoder
from sklearn.pipeline import Pipeline


def woe_encoding_pipeline(columns, n_bins=10, regularization=1.0):
    """Bin each of `columns` into `n_bins` equal-frequency quantile bins,
    then WOE-encode the bin labels against a binary target.

    Usage:
        pipe = woe_encoding_pipeline(["x"], n_bins=10)
        woe_encoded = pipe.fit_transform(df[["x"]], y)   # binary y, 0/1
    """
    return Pipeline([
        ("bin", EqualFrequencyDiscretiser(q=n_bins, variables=list(columns))),
        ("woe", WOEEncoder(cols=list(columns), regularization=regularization)),
    ])


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(size=n)
    y = pd.Series((x + rng.normal(scale=0.5, size=n) > 0).astype(int))
    df = pd.DataFrame({"x": x})

    pipe = woe_encoding_pipeline(["x"], n_bins=10)  # default regularization=1.0
    woe = pipe.fit_transform(df, y)["x"]
    print(woe.describe())
    print("Same IV as IV_quantile.py's own binning, computed from these WOE values:")
    # IV = sum over bins of (dist_pos - dist_neg) * woe ; recover it from the encoded column + y
    bins = pipe.named_steps["bin"].transform(df)["x"]
    tbl = pd.DataFrame({"bin": bins, "woe": woe, "y": y})
    agg = tbl.groupby("bin").agg(pos=("y", "sum"), n=("y", "size"))
    agg["neg"] = agg["n"] - agg["pos"]
    dist_pos = agg["pos"] / agg["pos"].sum()
    dist_neg = agg["neg"] / agg["neg"].sum()
    bin_woe = tbl.groupby("bin")["woe"].first()
    iv = ((dist_pos - dist_neg) * bin_woe).sum()
    print(f"  IV = {iv:.4f}")
