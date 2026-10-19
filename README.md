# dsutils

Personal data-science utilities that don't have a good home in an existing
maintained package. Started as 17 loose scripts
(https://github.com/lcrmorin/Python_DS_utilities); most of those turned out
to be already covered, exactly or better, by an existing package. This repo
keeps only the parts that survived that check.

See `existing_package_mapping.md` for the full script-by-script verification:
what each original script did, which package (if any) replaces it, and the
numbers proving equivalence (or the gap that means it doesn't). See
`annotated_originals/` for the original scripts with those findings as
inline comments.

## What's actually here

- `dsutils.dataviz` - `display_sbs`, a real importable version of the
  side-by-side pandas Styler display (itables doesn't do inline/Styler
  display), plus fold-size and percentogram plotting helpers.
- `dsutils.feature_interactions` - `run_screen`, eta-squared based screening
  for pairwise feature interactions (add/sub/mul/div) against one or more
  targets. Not covered by `sklearn.model_selection.permutation_test_score`
  (that's estimator-cv-bound, not a generic statistic).

- `dsutils.time_series` - `scan_trends`, trailing-window linear trend scanning
  with signed slope t-statistics, built on SciPy regression.

## What's deliberately not here

`reduce_memory`, `IV_quantile`, `Stability_Indicator` (PSI),
`better_pandas_describe`, `plot_feature_binning`, WOE encoding, and
correlated-feature reduction are all better served by an existing package
(`pandas.to_numeric`/`optbinning`/`evidently`/`skimpy` with caveats/
`feature-engine`+`category-encoders`/`feature-engine`). See
`existing_package_mapping.md` for the verification behind each call, and
`recipes/` for ready-to-use pipelines built on those packages.

## Install

```
pip install .
```

Optional extras:

```
pip install ".[recipes]"       # WOE encoding + correlation-cluster recipes
pip install ".[test]"          # everything needed to run the test suite
```

## Test

```
pip install ".[test]"
pytest
```

## Layout

```
src/dsutils/            installable package
recipes/                reference pipelines built on existing packages (not installed)
annotated_originals/    original 17 scripts + verified findings as comments
existing_package_mapping.md   the verification writeup
tests/                  pytest suite
```

## Trend scanning

```python
import numpy as np
from dsutils.time_series import scan_trends

rng = np.random.default_rng(42)
values = np.arange(72) * 0.2 + rng.normal(size=72)
scan = scan_trends(values, min_lookback=6, max_lookback=60)
best = scan.loc[scan.t_stat.abs().idxmax()]  # strongest trend in either direction
print(best)
```

Each candidate is a trailing window ending at the final observation. The returned
DataFrame contains the window's start/end labels, intercept, slope, slope standard
error, t-statistic and R-squared, indexed by lookback. Slopes are per observation;
Series timestamps are labels, not elapsed-time predictors. The intercept uses
x=0 at the beginning of each candidate window. Missing/infinite values are rejected.
Use `scan.t_stat.idxmax()` for the original script's largest signed t-statistic
ranking (which can still select a negative slope if all candidates slope down).
Ties select the shortest candidate. Constant windows use t=0 and R-squared=0;
nonconstant zero-standard-error fits use signed infinite t.

The statistic is descriptive: searching many overlapping windows and correlated
residuals make an ordinary t-test significance interpretation inappropriate.
The implementation delegates regression to
[`scipy.stats.linregress`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html);
it adds the trailing-window scan, tabular results and explicit edge-case handling.

Run `python examples/trend_scanning.py` for a seeded synthetic example, candidate
fits and the selected trend. It writes `trend_scanning.png`. Importing the module
has no plotting or random-data side effects.
