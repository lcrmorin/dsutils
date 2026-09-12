import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.core.base import PandasObject

def better_describe(df, values_list=[np.inf,-np.inf], prop=True):

    df_describe = df.describe(include='all')
    df_describe.loc['dtype'] = df.dtypes

    if prop:
        df_describe.loc['unique'] = np.round(df.nunique()/len(df),3)
        df_describe.loc['nan'] = np.round(df.isna().sum()/len(df),3)
    else:
        df_describe.loc['unique'] = df.nunique()
        df_describe.loc['nan'] = df.isna().sum()

    df_describe.loc['constant_diff'] = np.nan
    for c in df.columns:
        if is_numeric_dtype(df[c]):
            diffs = np.diff(np.unique(df[c]))
            min_diffs = min(diffs)
            if all([i.is_integer() for i in (diffs/min_diffs)]):
                df_describe.loc['constant_diff',c] = min_diffs

    for value in values_list:
        if prop:
            df_describe.loc[f'{str(value)}'] = np.round((df == value).sum()/len(df),3)
        else:
            df_describe.loc[f'{str(value)}'] = (df == value).sum()

    order = ['dtype','count','unique', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'constant_diff', 'nan'] + [f'{str(value)}' for value in values_list]

    return df_describe.loc[order]

PandasObject.better_describe = better_describe

# ---------------------------------------------------------------------------
# skimpy (pip install skimpy) covers part of this, verified by actually
# running `skimpy.skim(df)` side by side with this function, not just
# reading its README:
#
#   - Has: NA count, NA %, mean/sd/percentiles, dtype grouping, a mini
#     histogram per column (skimpy prints more than you compute, in that
#     one respect).
#   - Does NOT have, and there's no argument to turn on: a `constant_diff`
#     row (your GCD-of-unique-values check for detecting an implicit
#     step/scale in a column), or a configurable sentinel-value counter
#     (your `values_list=[np.inf, -np.inf]`, which you could point at any
#     value like -999).
#   - Actually crashes on the exact case this function is built for:
#     a column containing `inf`/`-inf` makes `skimpy.skim(df)` raise
#     `ValueError: autodetected range of [1.0, inf] is not finite` from
#     its internal histogram call. Reproduced live, not assumed - so
#     skimpy is not just "missing a feature" here, it's actively unusable
#     on data with the sentinel values this function is designed to flag.
#
# Net: skimpy is a good `df.describe()` replacement for the general case,
# but it cannot stand in for this function - keep `better_describe` for
# the inf/-inf + constant_diff checks, or feed skimpy already-cleaned
# (sentinel-replaced) data if you want its histograms on top.
# ---------------------------------------------------------------------------
