**Repo:** numpy/numpy
**Type:** Feature request (ENH)

## Title
Add a grouped covariance/correlation primitive (`grouped_cov` / `grouped_corrcoef`)

## Is your feature request related to a problem?

There's no numpy-level primitive for computing one aggregate per contiguous group of rows, without pulling in pandas' full labeling/alignment machinery. NEP 8 proposed close to this (`ufunc.reduceby`/`reducein`, scalar-per-group reductions over unique group ids or index-pair slices) - it's listed as Deferred, not rejected, with no successor. It only covers scalar-per-group output though; correlation/covariance per group needs a matrix output, which `reduceby`/`reducein` don't reach since `corrcoef` isn't a ufunc.

Ran into this computing grouped correlation matrices on panel data. Splitting by group and calling `np.corrcoef` per group beats `pandas.DataFrameGroupBy.corr()` once you're past a handful of columns (results identical, `np.allclose` checked every time):

```
30 cols,  500 groups x 400 rows: pandas 0.59s  vs  manual numpy 0.55s
100 cols, 500 groups x 400 rows: pandas 2.57s  vs  manual numpy 1.75s
30 cols,  5000 groups x 40 rows: pandas 1.60s  vs  manual numpy 1.68s
```

pandas does more per-group work as column count grows, and there's no numpy building block for "correlation matrix per contiguous group" - everyone doing this hand-rolls the same `np.split` + loop.

## Describe the solution you'd like

A group-boundary based primitive, `np.add.reduceat`-style but not limited to ufunc reduce - something like `np.group_apply(arr, group_boundaries, func)` for arbitrary per-group array-to-matrix functions, or more narrowly `np.grouped_cov`/`np.grouped_corrcoef(arr, group_boundaries)` as the concrete case people actually need.

## Describe alternatives you've considered

`numpy-groupies` covers scalar-per-group reductions, not a matrix output per group - different shape of problem, and third-party besides. Hand-rolling `np.split` + a loop works but doesn't parallelize well for many small groups - `joblib.Parallel(n_jobs=-1)` was 3x slower than serial with 500 groups, dispatch overhead dominating. rapidsai/cudf#8691 asks for the same grouped-correlation capability, open and unresolved - not just a one-off want on my end.
