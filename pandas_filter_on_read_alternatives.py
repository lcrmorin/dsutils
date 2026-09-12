"""Better alternatives to pandas_filter_on_read.py, verified against it.

Original script:

    iter_csv = pd.read_csv('file.csv', iterator=True, chunksize=1000)
    df = pd.concat([chunk[chunk['field'] > constant] for chunk in iter_csv])

This still reads and parses every row of the file - it just avoids holding
the *whole unfiltered* file in memory at once by processing it in pieces.
Real predicate pushdown means the engine skips irrelevant data before fully
parsing it. Two options that actually do that:
"""

import time

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.dataset as ds


# --- generate a test CSV, same shape of task as the original: filter field > constant ---

def make_test_csv(path: str, n_rows: int = 2_000_000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "field": rng.integers(0, 100, size=n_rows),
        "value": rng.normal(size=n_rows),
        "label": rng.choice(["a", "b", "c"], size=n_rows),
    })
    df.to_csv(path, index=False)


# --- the original approach ---

def original_pandas_chunked(path: str, constant: int) -> pd.DataFrame:
    iter_csv = pd.read_csv(path, iterator=True, chunksize=100_000)
    return pd.concat([chunk[chunk["field"] > constant] for chunk in iter_csv], ignore_index=True)


# --- option 1: polars lazy scan with real predicate pushdown ---

def polars_scan_filter(path: str, constant: int) -> pd.DataFrame:
    return (
        pl.scan_csv(path)
        .filter(pl.col("field") > constant)
        .collect()
        .to_pandas()
    )


# --- option 2: pyarrow dataset, if you want to stay pandas-adjacent without adding polars ---

def pyarrow_dataset_filter(path: str, constant: int) -> pd.DataFrame:
    dataset = ds.dataset(path, format="csv")
    table = dataset.to_table(filter=ds.field("field") > constant)
    return table.to_pandas()


if __name__ == "__main__":
    test_path = "/tmp/filter_on_read_test.csv"
    make_test_csv(test_path, n_rows=2_000_000)
    CONSTANT = 90  # selective filter: keeps ~10% of rows, the case pushdown helps most

    results = {}
    timings = {}
    for name, fn in [
        ("original (pandas, chunked)", original_pandas_chunked),
        ("polars.scan_csv (pushdown)", polars_scan_filter),
        ("pyarrow.dataset (pushdown)", pyarrow_dataset_filter),
    ]:
        t0 = time.perf_counter()
        out = fn(test_path, CONSTANT)
        elapsed = time.perf_counter() - t0
        results[name] = out.sort_values(["field", "value"]).reset_index(drop=True)
        timings[name] = elapsed
        print(f"{name}: {elapsed:.2f}s, {len(out):,} rows")

    baseline = results["original (pandas, chunked)"]
    for name, out in results.items():
        if name == "original (pandas, chunked)":
            continue
        match = (
            len(out) == len(baseline)
            and np.allclose(out["value"].values, baseline["value"].values)
            and (out["field"].values == baseline["field"].values).all()
        )
        print(f"{name} matches original output: {match}")

    fastest = min(timings, key=timings.get)
    speedup = timings["original (pandas, chunked)"] / timings[fastest]
    print(f"\nFastest: {fastest} ({speedup:.1f}x faster than the original chunked approach)")
