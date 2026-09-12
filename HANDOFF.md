# Project handoff: dsutils

Read this first if you're a fresh Claude session picking this up with no memory of prior conversation. It's a status snapshot, not a spec — the source repo and this package are the ground truth if anything here goes stale.

## What this project is

The user (lcrmorin) has a personal GitHub repo, [Python_DS_utilities](https://github.com/lcrmorin/Python_DS_utilities), of 17 loose, notebook-style Python scripts (data-science helpers: pandas utilities, binning/stability stats, a few plots, an XGBoost-to-SQL converter). They asked for it to be made "more professional and more useful, maybe a package."

Rather than packaging all 17 scripts as-is, the direction that was chosen (explicitly, after asking) is: **for each script, find out whether an existing, actively-maintained package already does it better** — and only keep/build what's genuinely not covered elsewhere. This repo (`dsutils`) is the small, deliberate remainder: things that earned their place after that check, built properly (tests, docstrings, real edge-case handling) rather than copy-pasted.

The user said they'll set up their own dedicated Python project/repo for this — this working copy is scratch state for continuing the analysis and code, meant to be dropped into that real project once it exists.

## Where things stand, function by function

Full detail and sources are in two docs at the repo root — **read `existing_package_mapping.md` before redoing any of this research**, it already answers "why doesn't X already exist as a package" for every one of the 17 original scripts, including specific GitHub issue links and maintenance-staleness checks (star counts, last-release dates, whether a small alternative package is abandoned) for pandas, scikit-learn, statsmodels, and a dozen smaller domain packages (probatus, optbinning, feature_engine, skimpy, jsharpe, scorecardpy, polars, itables, glances, etc.). Don't re-search this from scratch; extend it if the user asks about a script not yet covered there.

Short version of the verdicts:

- **Keep and build properly:** `percentogram_viz.py` → `dsutils.dataviz.percentogram` (done, see below). `xgboost_to_sql.py` → `dsutils.xgboost_sql` (done and actively being hardened, see below — this is the flagship piece the user specifically wants to maintain long-term).
- **Depend on an existing package instead** (retire the original script): `reduce_memory.py`, `better_pandas_describe.py`, `IV_quantile.py`, `Stability_Indicator.py`, `plot_feature_binning.py`, `AutoFE.py` (mostly — see surprise finding below), `pandas_filter_on_read.py`, `display_side_by_side.py`, `monitoring.sh`, `probabilistic_sharpe.py` (new finding: `jsharpe` covers this, reversing an earlier "keep it" call).
- **Drop entirely:** `data_viz.py`, `term_structure_viz.py`, `data_viz_venn.py` — thin wrappers over `seaborn`/`matplotlib_venn` with no added value. `plot_folds.py`, `numpy_groupy_corr.py` — low value, keep only if the user actually uses them.
- **Genuine surprise finding, worth knowing before touching `AutoFE.py` again:** its core row-wise-aggregate-feature functionality is already native in modern pandas — `df.agg(['mean', 'std', scipy.stats.skew, ...], axis=1)` works out of the box (confirmed empirically on pandas 3.0.2 in this sandbox). `feature_engine.creation.MathFeatures` is a sklearn-`Pipeline`-friendly wrapper around that, not a gap-filler. Don't reintroduce a hand-rolled version of this.

## `dsutils.dataviz` — done, no open work

`src/dsutils/dataviz/percentogram.py`. One function, `percentogram(series, num_bins=10, ax=None, **bar_kwargs)`, drawing an equal-mass (quantile-width) histogram. Fixed two real bugs versus the original script: `num_bins` was silently ignored (always used 10), and it forced `plt.show()` making it untestable/unusable outside a live notebook. 8 tests in `tests/test_percentogram.py`, all passing. The user said this one's fine as-is — no further work expected here unless asked.

## `dsutils.xgboost_sql` — active work, most important part of this handoff

`src/dsutils/xgboost_sql/converter.py` (implementation) + `src/dsutils/xgboost_sql/__init__.py` (re-exports `xgboost_to_sql`, `xgboost_to_sql_multiclass`, `prepare_df_for_duckdb`). Converts a trained XGBoost `Booster` into a SQL `CASE WHEN` expression for in-database scoring, with float32-precision-correct thresholds and categorical-split support. This is the one function of the original 17 with no good existing substitute (closest prior art — `gbm2sql`, `sqlgbm`, R-only `xgb2sql`, `m2cgen` which doesn't target SQL at all — are all small, stale, or explicitly not production-ready; see `existing_package_mapping.md` for specifics). The user explicitly wants to maintain this one for real production use and asked for serious test-hardening.

**Read `TESTING_PLAN.md` before changing this file.** It has the full list of what's been verified and what's still open. Highlights:

Fixed real bugs this session:
- Unescaped single quotes in categorical SQL literals (`'{v}'` → injection/breakage risk for any label like "O'Brien"). Fixed, tested.
- `booster="dart"` silently returned wrong predictions (verified empirically: summed-tree SQL gave ~0.007 vs the model's real ~0.25). Now raises `NotImplementedError` immediately rather than a wrong number.
- `booster="gblinear"` now rejected immediately (not a tree model).
- Objectives beyond binary:logistic/reg:squarederror were unsafe — `count:poisson`/`reg:gamma`-style models called with the old defaults silently returned a raw margin as if it were a real prediction. Added an empirically-verified link-function table (`identity`/`logistic`/`exp`) with auto-detection from the model's objective; unrecognized objectives now raise instead of guessing. Real `exp`-link support was added and tested (not just blocked).
- `base_score` exactly 0/1 (logit blows up) and `base_score <= 0` (log undefined) now raise clearly instead of a silent wrong fallback.
- `num_class` mismatch in `xgboost_to_sql_multiclass` now raises instead of silently misassigning trees to classes.
- Confirmed (didn't need a code change, but was worth verifying rather than assuming): `num_parallel_tree > 1` (random-forest mode) needs **no** special handling — XGBoost's own leaf values already account for it. Locked in with a regression test.

Real, unfixable-in-general limitation found and documented (not a bug in this code):
- **SQLite cannot be a target engine.** `CAST(x AS FLOAT)` is a no-op there (no true 4-byte float type), so the float32-precision trick this tool depends on breaks — confirmed empirically, ~7% of rows misrouted, up to 0.43 absolute error. Test asserts this failure mode explicitly rather than hiding it.

**The single most important open item:** the module's docstring claims "works on MySQL (FLOAT) and PostgreSQL (REAL)" — **that has never actually been verified against either engine**, only DuckDB (and now SQLite, which is documented as broken). This sandbox has no Docker daemon (`docker ps` fails — no daemon socket), so a real Postgres/MySQL container couldn't be spun up here. **If you have Docker available, this is the top priority**: write a test (or a GitHub Actions workflow using the `postgres:`/`mysql:` service-container pattern) that trains a model, generates SQL with `float_type="REAL"` (Postgres) or `float_type="FLOAT"` (MySQL), runs it against a real instance, and compares to `Booster.predict()`. Given what SQLite revealed, do not assume this claim is true until it's actually checked — verify whether Postgres's `REAL` and MySQL's `FLOAT` are true 4-byte types (they should be, unlike SQLite, but confirm empirically rather than assuming).

Other open items from `TESTING_PLAN.md`, roughly in priority order after Postgres/MySQL: unseen-category handling at inference time (recommend NULL-mapping as the documented safe practice rather than testing undefined behavior); property-based testing with `hypothesis` for edge cases hand-written tests won't think of; SQL length/nesting limits at scale (500+ trees, depth 10+); `reg:tweedie` and `multi:softmax` are in the code's allow-list/reasoning but not yet locked in with their own tests.

28 tests currently pass (`pytest tests/test_xgboost_sql.py -v`, ~90 seconds — most of the time is XGBoost training, not SQL generation).

## Running things

```
pip install --break-system-packages -e ".[test]"   # or drop --break-system-packages in a venv
pytest tests/ -v
```

Dependencies needed for the test suite: `xgboost`, `duckdb`, `pytest` (none of these are in a bare sandbox by default — install them). `sqlite3` is stdlib. Real Postgres/MySQL testing would additionally need `psycopg2`/`pymysql` or similar plus a running server — not currently set up anywhere.

## Things this session could NOT do — be upfront about these rather than re-attempting them blindly

- **No push access to the user's GitHub repo** and no `gh` CLI auth in this sandbox. Everything has been handed to the user as files/zips, not pushed. Don't assume you can push either unless the environment has changed.
- **No Docker daemon** in this sandbox (checked: `docker` binary exists but `docker ps` fails to find a socket). Blocks real Postgres/MySQL verification for `xgboost_sql` — see above.
- **GitHub's API (`api.github.com`) is blocked** by this sandbox's proxy for unauthenticated calls ("GitHub access to this repository is not enabled for this session"); some GitHub HTML pages are also blocked by robots.txt for automated fetches (e.g. `/commits` pages). Star counts / release dates in `existing_package_mapping.md` were gathered via PyPI's JSON API (`https://pypi.org/pypi/<name>/json`, reliable and unblocked) plus `WebFetch`/`WebSearch` on plain repo pages where that worked — reuse that approach rather than trying `api.github.com` again.

## What the user said to do next

Their last message: focus on the *other* functions (not `xgboost_sql`/`dataviz`, which are considered settled), specifically digging into why each gap isn't already merged into pandas/scikit-learn/statsmodels, with citations to real issues where they exist, and naming a "big package" alternative target for each. That research is now in `existing_package_mapping.md` under "Why not just in pandas / scikit-learn / statsmodels?" — if the user asks for more of this, that's the section to extend, following the same pattern (search real issue trackers, don't guess; PyPI JSON API for release freshness; be explicit when the honest answer is "no big-package target exists and here's why").
