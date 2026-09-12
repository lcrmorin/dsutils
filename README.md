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
- `dsutils.xgboost_sql` - XGBoost-to-SQL export (dialect-aware CASE/IF trees,
  float32-exact literal matching, categorical splits). Not covered by
  `m2cgen` (no SQL target) or the narrow single-purpose SQL exporters on
  PyPI.

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
pip install ".[xgboost-sql]"   # XGBoost -> SQL export
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
issue_drafts/           upstream issues/PRs drafted from genuine gaps found here
```
