**Repo:** numpy/numpy
**Type:** Feature request (ENH)

## Title
`histogram_bin_edges`: add an equal-frequency (quantile) binning strategy

## Is your feature request related to a problem?

`np.histogram_bin_edges(x, bins=...)` supports `'auto'`, `'fd'`, `'doane'`, `'scott'`, `'stone'`, `'rice'`, `'sturges'`, `'sqrt'` - all equal-*width* strategies (they pick a bin width/count, then space edges evenly across the data's range). There's no built-in equal-*frequency* strategy (edges placed so each bin holds roughly the same number of observations), even though it's one of the two textbook-standard binning approaches and is what you reach for whenever the data is skewed and equal-width bins would leave most of the mass in one or two bins - a genuinely common histogram to want, not a niche one.

Today it has to be hand-rolled: `np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))`, plus your own handling of the duplicate-edge case when the data has enough repeated values that some requested quantiles land in the same spot.

## Describe the solution you'd like

`np.histogram_bin_edges(x, bins='quantile')` (naming open to bikeshedding - `'equal_frequency'` also reads clearly), using the number of bins from a paired `range`/`bins=(strategy, n)` argument or a sensible default, with `np.unique` applied to the resulting edges so duplicate quantiles collapse into fewer, wider bins rather than erroring or producing zero-width bins.

## Describe alternatives you've considered

`pandas.qcut` does the equal-frequency binning itself, but returns pandas-specific `Categorical` bin labels tied to a Series, not raw histogram bin edges compatible with `np.histogram`/`plt.hist` - a different-shaped output for a different purpose (labeling data by bin vs. building a histogram). There's no numpy-native way to get equal-frequency edges as plain floats.

## Additional context

Happy to submit a PR - the core computation is a few lines (see the "solution" section above), the main design decision is naming and how to specify bin count for this strategy consistently with the existing string-strategy API.
