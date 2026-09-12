"""Hardened version of reduce_memory.py - same approach, bugs fixed.

Two real bugs found and fixed (see existing_package_mapping.md for the
verification behind each):

1. `col_type != 'object'` breaks on pandas >= 3.0's default string dtype
   (columns of Python strings are no longer `object` by default - they're
   `str`/StringDtype). Fixed by checking `is_numeric_dtype` directly
   instead of excluding 'object' by name.
2. `c_max < np.iinfo(...).max` / `c_max < np.finfo(...).max` use a strict
   bound, so a value exactly at the dtype's max gets bumped to the next
   size up unnecessarily. Fixed to `<=`/`>=` (the same inclusive bound
   `pd.to_numeric(downcast=...)` uses).

Kept float16 as a downcast target (unlike `pd.to_numeric`, which stops at
float32) since that's the original script's intended behavior - pass
`allow_float16=False` if you want pandas' more conservative behavior.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype
from pandas.core.base import PandasObject


def reduce_memory(df, threshold_str=0.5, allow_float16=True, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min >= np.iinfo(np.int64).min and c_max <= np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif is_float_dtype(col_type):
                if allow_float16 and c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type == 'object' or col_type == 'category' or str(col_type) == 'str':
            if col_type == 'category':
                continue
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_total_values and num_unique_values / num_total_values < threshold_str:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df


PandasObject.reduce_memory = reduce_memory


if __name__ == "__main__":
    # regression test for both fixes, against the ORIGINAL (buggy) function
    import warnings

    def reduce_memory_original(df, threshold_str=0.5):
        for col in df.columns:
            col_type = df[col].dtype
            if col_type != 'object' and col_type != 'category':
                c_min = df[col].min(); c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            elif col_type == 'object':
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < threshold_str:
                    df[col] = df[col].astype('category')
        return df

    # 1. boundary case: a column whose max is exactly int8's max (127)
    df_boundary_a = pd.DataFrame({"x": np.array([0, 50, 127], dtype=np.int64)})
    df_boundary_b = df_boundary_a.copy()
    reduce_memory_original(df_boundary_a)
    reduce_memory(df_boundary_b, verbose=False)
    print("boundary fix: original ->", df_boundary_a["x"].dtype,
          "| fixed ->", df_boundary_b["x"].dtype)
    assert df_boundary_b["x"].dtype == np.int8 and df_boundary_a["x"].dtype == np.int16

    # 2. string-dtype case: only reproducible when pandas' default string
    # dtype is active (pandas >= 3.0). Skips cleanly on older pandas.
    if pd.__version__ >= "3":
        df_str = pd.DataFrame({"s": ["a", "b", "c"]})
        try:
            reduce_memory_original(df_str.copy())
            print("string-dtype fix: original did NOT crash (unexpected on this pandas version)")
        except Exception as e:
            print(f"string-dtype fix: original crashes as expected ({type(e).__name__})")
        reduce_memory(df_str.copy(), verbose=False)
        print("string-dtype fix: hardened version handles it fine")

    print("All checks passed.")
