**Repo:** pandas-dev/pandas
**Type:** Feature request (ENH)

**DUPLICATE CHECK - already effectively declined, don't file this as written.** [pandas#21689](https://github.com/pandas-dev/pandas/issues/21689) asked for just a missing-value count in `describe()` (narrower than this draft) and a maintainer (jreback) responded that the team "don't want to add things to describe, as bespoke additions are easy for users to implement on their own." That's a direct precedent against this exact shape of request - filing a broader version of the same ask isn't likely to land differently. If this is still worth raising, it'd need a different angle than "add more columns to describe()" - e.g. proposing a *separate* `df.profile()`/`df.summary()` method instead of extending `describe()` itself, since the objection was specifically about scope-creeping `describe()`, not about the underlying columns being uninteresting.

## Title
ENH: optional `extra` columns for `DataFrame.describe()` - dtype, % unique, % missing, and a sentinel-value counter

## Is your feature request related to a problem?

`df.describe(include='all')` gives count/mean/std/percentiles/unique/top/freq, but leaves out a few columns that are almost always the next thing anyone checks by hand right after calling it:

- the column's `dtype` (have to `pd.concat([df.describe(), df.dtypes.rename('dtype').to_frame().T])` yourself)
- proportion unique / proportion missing (as a fraction of row count, not raw counts - `nunique()/len(df)` and `isna().mean()`)
- a count of how many values equal a given sentinel (`np.inf`, `-np.inf`, or a domain-specific value like `-999`) - useful because sentinel/placeholder values are a common way missingness gets silently encoded in numeric columns, and `describe()`'s mean/std get quietly distorted by them without this check

None of these needs new statistical logic - each is a one-liner - but doing all of them by hand every time is the actual friction, and there's no existing pandas issue asking for this (checked).

## Describe the solution you'd like

```python
df.describe(include='all', extra=['dtype', 'unique_pct', 'missing_pct'], sentinels=[np.inf, -np.inf])
```

appending rows to the existing `describe()` output rather than replacing any of it, so it stays backward compatible.

## Describe alternatives you've considered

This is squarely in the "one-call EDA summary" niche that a whole separate ecosystem already covers instead of pandas core: `ydata-profiling`, `sweetviz`, `dtale`, `skimpy`, `klib`. Checked `skimpy` specifically since it's the closest in spirit (NA%, mean/sd/percentiles, a mini histogram) - it doesn't have the sentinel-count or dtype-row either, and it actually crashes (`ValueError: autodetected range ... is not finite`) on a column containing `inf`, which is exactly the case this ask is meant to catch. So there's a real, verified gap even against the best existing lightweight option, not just a preference for pandas-native.

## Additional context

Understand pandas has deliberately kept `.describe()` narrow for years while this ecosystem grew up around it, so this may be a "please use skimpy/ydata-profiling instead" close - filing mainly to get that answer on record, and because the sentinel-count feature specifically doesn't seem to exist anywhere I could find, pandas-native or not.
