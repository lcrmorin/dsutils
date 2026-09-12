**Repo:** pandas-dev/pandas
**Type:** Feature request (ENH)

**DUPLICATE CHECK - do not file as a new issue.** [pandas#40692](https://github.com/pandas-dev/pandas/issues/40692), "Function to optimise dataframe (efficient datatypes)," already asked for essentially the same `pd.optimize(df)`-style helper. It's closed, labeled "Closing Candidate," with no maintainer discussion recorded. If you still want to pursue this, the move is to comment on #40692 (referencing the two concrete bugs below as new, concrete motivation a closed feature request didn't originally have) rather than opening a duplicate - a second identical ask without new information is likely to get closed the same way. The bug-report angle (item 1 below) is still worth raising on its own if you want it - it's distinct from the feature request and nothing found addresses it directly (see `pandas_string_dtype_object_check_footgun.md`).

## Title
ENH: `DataFrame.optimize_dtypes()` — downcast every column's dtype in place, with a memory-usage report

## Is your feature request related to a problem?

There's no single pandas call that downcasts every column of a DataFrame to the smallest safe numeric dtype and converts low-cardinality object/string columns to `category`. This is one of the most common "getting started with a large CSV" recipes (Kaggle notebooks alone have hundreds of nearly-identical hand-rolled copies of it), and it's easy to get subtly wrong - I found two real bugs in one such copy while comparing it against `pd.to_numeric(..., downcast=...)`:

1. Using `col.dtype != 'object'` to mean "is numeric" silently breaks under `future.infer_string` / pandas 3.0's default string dtype, since string columns are no longer `object` by default (already tracked as a breaking-change category in #59328, but this specific "numeric-vs-string branch selection" failure mode isn't called out there and is easy to hit).
2. Off-by-one bound checks (`c_max < np.iinfo(np.int8).max` instead of `<=`) push values that fit exactly at a dtype's max into the next size up.

`pd.to_numeric(col, downcast='integer'/'float')` already solves the per-column numeric part correctly and robustly (verified: matches a corrected hand-rolled loop exactly, modulo pandas' deliberate choice to never downcast to float16). What's missing is:
- a whole-DataFrame convenience wrapper that applies it column-by-column
- the low-cardinality-object-to-category step
- the optional start/end memory-usage report that makes this genuinely useful interactively (this is the part every hand-rolled version reimplements)

## Describe the solution you'd like

Something like:

```python
df, report = df.optimize_dtypes(category_threshold=0.5, allow_float16=False, verbose=True)
# Memory usage of dataframe is 125.34 MB
# Memory usage after optimization is: 31.20 MB
# Decreased by 75.1%
```

Internally this is a thin, well-tested wrapper around `pd.to_numeric(..., downcast=...)` per numeric column and `.astype('category')` per low-cardinality object/string column, using `is_numeric_dtype`/`is_string_dtype` (not `== 'object'`) so it doesn't inherit the string-dtype pitfall above.

## Describe alternatives you've considered

- Third-party packages (`pandas-downcast`, `dtype-diet`) do roughly this, but both have had no release in 3+ years.
- `df.convert_dtypes(dtype_backend="pyarrow")` is the closest core-pandas feature, but it solves a different problem: verified it does NOT shrink numeric column width at all (an int64 column holding 0-99 stayed exactly 8000 bytes for 1000 rows, before and after) - its real benefit is on string columns (11x smaller in the same test), which is a different, complementary mechanism (Arrow-backed dtype storage) rather than per-column int/float downcasting.

## Additional context

Happy to submit a PR with the wrapper above plus tests if this is a direction pandas wants; wanted to check first since a fully-general "shrink my whole DataFrame" scope might be considered out of bounds for pandas core (similar to why some other close cousins of this ask haven't landed).
