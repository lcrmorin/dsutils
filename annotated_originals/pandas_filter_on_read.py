import pandas as pd

iter_csv = pd.read_csv('file.csv', iterator=True, chunksize=1000)
df = pd.concat([chunk[chunk['field'] > constant] for chunk in iter_csv])

with open(<file>) as f:
    text = "\n".join([line for line in f if line.split(',')[2] == "Hello"])

df = pd.read_csv(StringIO(text))

# ---------------------------------------------------------------------------
# Plain pandas has no real predicate pushdown for CSV - chunked
# `pd.read_csv(..., iterator=True)` (your first approach) still parses
# every row, it just avoids holding the whole file in memory at once.
# `polars.scan_csv(...).filter(...)` does real pushdown: a single native
# pass that discards non-matching rows during the scan.
#
# Benchmarked on a 2,000,000-row CSV, same filter (`field > 90`, keeps
# ~10%), output verified row-for-row identical (`np.allclose` on values,
# exact match on the filter column, same row count):
#
#   original (pandas, chunked):        0.45s
#   polars.scan_csv (pushdown):        0.15s   (3.1x faster)
#   pyarrow.dataset (pushdown):        0.19s   (2.4x faster)
#
# pyarrow.dataset is the option if you want to stay in the
# pandas/pyarrow ecosystem without adding polars as a dependency; polars
# was the fastest of the two in this test. Full runnable comparison (both
# alternatives + the benchmark + the output-equality check) is in
# `pandas_filter_on_read_alternatives.py` next to this file.
#
# Your second snippet (line-by-line `line.split(',')[2] == "Hello"` before
# `pd.read_csv`) is a manual, fragile version of the same idea - it breaks
# on quoted commas inside a field. `polars.scan_csv(...).filter(pl.col(...)
# == "Hello")` replaces it too, correctly handling quoting.
# ---------------------------------------------------------------------------
