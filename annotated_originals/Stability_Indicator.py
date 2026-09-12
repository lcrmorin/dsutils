import numpy as np, pandas as pd

Base = pd.Series([1,2,3,1,2,3,1,2])
New = pd.Series([1,2,3,1,2,3])

def PSI_indicator(Base, New, Verbose = False):

    Base_counts = Base.value_counts().sort_index()
    New_counts = New.value_counts().sort_index()

    Base_props = Base_counts/Base_counts.sum()
    New_props = New_counts/New_counts.sum()

    assert (Base_props.index == New_props.index).all()

    diff = New_props - Base_props
    var = New_props / Base_props - 1
    log_rel = np.log(New_props / Base_props)
    PSI = diff * log_rel

    res = pd.DataFrame({
    'base count':Base_counts.values,
    'new count':New_counts.values,
    'base prop':Base_props.values,
    'new prop':New_props.values,
    'diff':diff.values,
    'var':var.values,
    'log_rel':log_rel.values,
    'PSI':PSI.values
    })


    fmt = {
    'base count':'{:}',
    'new count':'{:}',
    'base prop':'{:.2%}',
    'new prop':'{:.2%}',
    'diff':'{:.2%}',
    'var':'{:.2%}',
    'log_rel':'{:.2%}',
    'PSI':'{:.2%}'
    }

    if Verbose:
        display(res.style.format(fmt))

    return np.mean(PSI)

PSI_indicator(Base, New, Verbose = True)

# ---------------------------------------------------------------------------
# Correction to earlier advice: probatus does NOT have PSI. Checked by
# importing it (`probatus` 3.1.5, the current PyPI release) and walking its
# actual submodules - there is no `stat_tests`/PSI/distribution-comparison
# module at all anymore; probatus 3.x only ships feature_elimination,
# interpret, and sample_similarity. Whatever PSI support it may have had in
# an old 1.x/2.x release is gone in what `pip install probatus` gives you
# today, so don't add it for this.
#
# `evidently` (pip install evidently) does have it, and its formula is
# numerically identical to yours once you account for one difference -
# verified by calling its internal PSI function directly on your own
# `Base`/`New` example (not synthetic data):
#
#   your PSI_indicator(Base, New)                     -> 0.011262919669671231
#   evidently's psi(Base, New, categorical) SUM        -> 0.033788759009013694
#   evidently's psi(...) / n_categories (their sum,
#     divided the same way you take a mean)            -> 0.011262919669671231  (exact match)
#
# So the real difference isn't the package, it's the aggregation: the
# standard PSI (and evidently's) SUMS the per-category terms; your
# function takes the MEAN (divides by the number of categories instead of
# leaving it summed). That's a nonstandard normalization worth knowing
# about even if you keep this function - a 3-category PSI from this code
# is not comparable to a 5-category PSI from anywhere else at the same
# "meaning" of PSI.
#
# evidently is also more robust on one point your `assert` isn't: it fills
# a category that's missing from Base or New with a near-zero count
# instead of raising `AssertionError` when Base_props.index != New_props.index.
# ---------------------------------------------------------------------------
