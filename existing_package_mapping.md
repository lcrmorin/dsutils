## Follow-up on the suggested improvements (WOE encoding, correlation prefilter)

Two of the "new functions" I suggested turned out to already be fully covered by packages already in play here - checked before writing anything new, so nothing got built that duplicates them:

- **WOE-encoding transform**: `feature_engine.discretisation.EqualFrequencyDiscretiser` + `category_encoders.WOEEncoder`, chained in a `sklearn.pipeline.Pipeline`. Verified the discretiser reproduces `IV_quantile.py`'s own quantile bins exactly (same row -> same bin index, not just same bin count), and `WOEEncoder`'s formula matches a hand-rolled per-bin WOE calculation exactly when its `regularization=0.0` - it's the identical formula plus optional additive smoothing to avoid the divide-by-zero a hand-rolled version hits on a pure bin. Recipe + 3 passing tests: `recipes/woe_encoding.py`, `tests/test_woe_recipe.py`.
- **Correlation-cluster feature reducer**: you asked for a scikit-learn transformer "if it doesn't exist yet" - it does. `feature_engine.selection.SmartCorrelatedSelection` (and the simpler `DropCorrelatedFeatures`) are real `BaseEstimator`/`TransformerMixin` subclasses - checked `isinstance()` directly, not just the docs - that cluster correlated features above a threshold and keep the best one per cluster by a chosen criterion (variance, cardinality, or a model's predictive score). Verified on synthetic near-duplicate data: correctly finds `{a, b}` as one cluster, drops one, keeps the independent column `c`. Recipe + 3 passing tests: `recipes/correlation_prefilter.py`, `tests/test_correlation_prefilter.py`. Use this ahead of `feature_interactions.run_screen` to cut its O(k^2) pair count.

On the other two: the time-series CV splitter is what you're already building, so nothing added there. On the drift report - it would genuinely just be a thin loop over `Stability_Indicator`/`evidently`'s PSI per column plus a sort, no new math - not built since the value of that wrapper versus just calling PSI in a loop yourself wasn't clearly worth it to you. The permutation-significance function is about `feature_interactions.run_screen` specifically: screening every feature pair x operator x target produces a lot of comparisons (thousands, easily), and some will show a large "gap" (the sign of a real interaction) purely from sampling noise even when there's no real interaction - a permutation-based null tells you what gap size is expected from noise alone, so you can tell which top rows in the output are likely real versus which are false positives from testing that many pairs at once. Not built yet - only worth it once you're running `run_screen` on a real feature set large enough for that multiple-testing effect to matter.

## This round's additions

- **`annotated_originals/reduce_memory_v2.py`** — your own function, hardened: fixes the pandas-3.x string-dtype crash and the off-by-one bound (both verified with a regression test against the original buggy version, in the file's own `__main__`).
- **`issue_drafts/`** — five ready-to-file GitHub issue drafts: a pandas feature request for a built-in dtype-downcast helper, a pandas feature request for `.describe()` extensions, a pandas performance report for `groupby().corr()` scaling with column count (backed by the benchmark below), and two scikit-learn feature requests (PSI, Information Value). None of these have been filed — no push/GitHub-issue access from this session — they're text, ready for you to paste in if you want to.
- **`numpy_groupy_corr.py` verdict reversed** — see the dedicated section below. Kept, not dropped; a column-index API was added alongside the original.
- **`src/dsutils/dataviz/side_by_side.py`** — `display_side_by_side.py` moved in as a real importable function (itables doesn't replace it, see prior finding).
- **`src/dsutils/feature_interactions/`** — new module built from a function you supplied (`run_screen`), with the missing pieces implemented and verified (eta-squared checked against `statsmodels`' one-way ANOVA; interaction-detection checked against a synthetic case with a known multiplicative interaction). See its own section below.

# Where these 17 scripts already exist elsewhere

For each script: what it does, the closest existing package, and a verdict. "Depend on it" means retire the script and `pip install` the alternative. "Keep / contribute" means nothing off-the-shelf covers it well enough to replace it.

**Jump to:** [why these gaps aren't in pandas / scikit-learn / statsmodels themselves](#why-not-just-in-pandas--scikit-learn--statsmodels).

Checked via GitHub + PyPI release metadata (today is 2026-09-12).

| Script | What it does | Target package | Maturity | Open issue upstream for your gap? | Verdict |
|---|---|---|---|---|---|
| `reduce_memory.py` | Downcasts int/float columns, casts low-cardinality object columns to `category` | Native pandas: `pd.to_numeric(col, downcast=...)` (third-party `pandas-downcast`/`dtype-diet` are both stale, see below) | Standalone packages both **stale — 3+ years since last release**; `pd.to_numeric` itself is core pandas, always current | No open pandas issue for this exact helper — doesn't fit pandas' current direction | **Real bug found on current pandas (3.0.2): fix `col_type != 'object'` before anything else** — it silently breaks on pandas 3.x's new default string dtype (see Follow-up findings). Beyond that, `pd.to_numeric(downcast=...)` covers the numeric half natively; keep your own loop for the category-cast half |
| `better_pandas_describe.py` | `.describe()` plus dtype, % unique, % NaN, constant-diff detection, sentinel-value counts | `skimpy` (aeturrell/skimpy — mind the name collision with an unrelated EPFL package) | Medium, single maintainer, actively developed, released 2026-06-04 (~3 months ago) | None found for `constant_diff` / sentinel-count columns — would need to file one | Depend on it for the general case; keep those two columns as a small local add-on |
| `IV_quantile.py` | Quantile-bins a continuous feature, computes Information Value vs. a binary target | `optbinning` | Medium-big, single main author, de facto standard in credit-scoring; released 2025-10-26 (~11 months ago) | N/A — IV is a shipped core feature | Depend on it |
| `Stability_Indicator.py` | Population Stability Index (PSI) with formatted breakdown table | ~~`probatus.stat_tests.PSI`~~ **correction below — use `evidently` instead** | `evidently`: big, actively maintained (v0.7.23, current release) | N/A, PSI is a shipped core feature | Depend on `evidently`'s PSI — verified numerically identical once you account for sum-vs-mean (see Follow-up findings) |
| `plot_feature_binning.py` | Bins a feature into quantiles, plots stacked class counts + target-rate line | `scorecardpy.woebin_plot`, or `optbinning`'s own binning-table plot | Medium-big by stars (796/315 forks) but **PyPI package stale since 2023-07-29** (~3 years); GitHub shows ongoing commits/issues but I couldn't confirm the latest commit date (blocked from automated access) | N/A | Verify scorecardpy's actual freshness yourself, or just lean on `optbinning`'s plot instead since you already depend on it for IV |
| `AutoFE.py` | Row-wise aggregate features (mean, std, skew, kurtosis, quantiles, mode, per-value counts) | `feature_engine.creation.MathFeatures` — **but see finding below, most of this is already native pandas** | Medium-big, sklearn-compatible, "Train In Data" ecosystem, released 2026-02-27 (~7 months ago) | None found for the per-value `count_i` tally — different kind of feature (categorical tally vs. numeric aggregate), likely out of scope | Use `df.agg([...], axis=1)` directly (native pandas, see below) for the aggregates; keep `count_i` as a one-liner; `feature_engine` only useful if you need an sklearn-`Pipeline`-compatible wrapper |
| `pandas_filter_on_read.py` | Filters rows while reading a large CSV in chunks | `polars.scan_csv(...).filter(...)` | Huge, extremely active, continuous releases | N/A (see below — pandas itself declined this) | Depend on it — real predicate pushdown, not just redundant |
| `display_side_by_side.py` | Renders several styled DataFrames side by side in a notebook | `itables` | Medium, single maintainer (also maintains `jupytext`), well-integrated in Jupyter/marimo, released 2026-07-22 (~7 weeks ago) | N/A — not a library concern, see below | Depend on it |
| `monitoring.sh` | Shell loop logging `free -m` every 60s | `glances` | Big, broad community, released 2026-08-01 (~6 weeks ago) | N/A | Depend on it. (`psrecord` is the tempting small alternative but stale since 2024-04-26, and the wrong shape anyway — per-process, your script is system-wide) |
| `plot_folds.py`, `numpy_groupy_corr.py` | CV-fold size chart; faster groupby-corr | `plot_folds.py` moved into `dataviz/fold_sizes.py`; `numpy_groupy_corr.py` — **verdict corrected, see below** | N/A | N/A | `plot_folds.py`: kept, now in dataviz (no scikit-learn equivalent — it's specific to your fold structure). `numpy_groupy_corr.py`: **kept, not dropped** — same values as `df.groupby('g')[cols].corr()` always, but genuinely faster once column count grows (benchmarked, see Follow-up findings); added a column-index-based API alongside the original |
| `data_viz.py`, `term_structure_viz.py`, `data_viz_venn.py` | Direct `seaborn`/`matplotlib_venn` calls with your default kwargs | The libraries themselves | N/A | N/A | Drop — not utilities, just usage examples |
| `percentogram_viz.py` | Equal-mass (quantile-width) histogram | None found | — | — | **Keep.** Now built into `dsutils.dataviz.percentogram` |
| `probabilistic_sharpe.py` | Probabilistic Sharpe Ratio + min track record length | `jsharpe` — **new finding, reverses my earlier answer** | Small but actively maintained: 22 stars, 430+ commits, CI + dependency bot, released 2026-07-09 (~2 months ago) | N/A | Depend on it instead of keeping your own — see below |
| `xgboost_to_sql.py` | XGBoost → SQL `CASE WHEN`, float32-correct thresholds, categorical + missing support | No good match — see below | — | — | **Moved out** — now maintained as its own project, not part of this package |

**Correction (verified by actually importing it):** `probatus` does **not** have PSI. I originally wrote it did, with a specific "corporate-backed, released 4 days ago" claim — that was wrong; I did not check hard enough the first time. `pip show probatus` gives version 3.1.5, and walking its actual submodules shows only `feature_elimination`, `interpret`, and `sample_similarity` — no `stat_tests`, no PSI, nothing distribution-comparison-related survives in the current release. `evidently` is the corrected recommendation for PSI (see Follow-up findings below for the numeric verification).

**Bottom line on maturity:** `optbinning`, `feature_engine`, `itables`, `polars`, `glances`, `evidently`, and `skimpy` are all genuinely actively maintained, safe to depend on. The one place where I initially pointed you at a "big package" that turned out to be **stale on both sides** is `reduce_memory.py` — neither `pandas-downcast` nor `dtype-diet` has shipped in 3+ years, and there's no actively maintained direct replacement, only pandas' own (structurally different) pyarrow-backed dtypes. `scorecardpy` is a "verify before you depend" case, not a clean green light.

## New finding since the first pass: `probabilistic_sharpe.py`

I previously said "keep this yourself, there's a real gap." I was wrong to stop searching there — [`jsharpe`](https://github.com/tschm/jsharpe) is a small but **actively maintained** package (PyPI release 2026-07-09, ~2 months ago; 22 stars but 430+ commits, CI, dependency-update bot configured) that does exactly Probabilistic Sharpe Ratio + minimum track record length, plus more (FDR/FWER multiple-testing corrections you don't have). Unless you specifically want to own this one too, I'd depend on `jsharpe` instead of keeping your own copy. `quantstats` (the big, actively-maintained portfolio package, last release 2026-01-13) still doesn't have PSR/DSR built in, for context — but a smaller purpose-built package already fills that gap, so there's no real reason to maintain your own version.

## `xgboost_to_sql.py` — why "keep" holds up

Closest prior art: [gbm2sql](https://github.com/crst/gbm2sql) (4 stars, 1 commit — a demo snapshot, not a maintained tool), [sqlgbm](https://github.com/mattismegevand/sqlgbm) (2 stars, README literally says "not ready for production use yet"), [xgb2sql](https://github.com/chengjunhou/xgb2sql) (R only). The general-purpose model-export tool [m2cgen](https://github.com/BayesWitnesses/m2cgen) has 3,000 stars but hasn't released since **April 2022** and doesn't target SQL as a language anyway. Nothing here is a real substitute, but this module now lives in its own separately-maintained project rather than in dsutils.

## Why not just in pandas / scikit-learn / statsmodels?

You asked specifically about the three foundational packages: is each gap a known open issue there, and why hasn't it been merged. Checked against real GitHub issues, not guessed.

### `AutoFE.py` (row-wise aggregate features) — mostly already IN pandas, corrected finding

I initially pointed this at `feature_engine.creation.MathFeatures`. Digging into pandas' own history changed the answer: [pandas#16679](https://github.com/pandas-dev/pandas/issues/16679) ("agg method with list of functions does not work with axis=1") was filed, accepted, and **fixed in pandas core** (PR #21224). I confirmed on the pandas version in this sandbox (3.0.2) that this now just works natively:

```python
df.agg(['mean', 'std', scipy.stats.skew, scipy.stats.kurtosis], axis=1)
```

That's most of what `AutoFE.py` does, in base pandas, no extra dependency at all. **Big package target: pandas itself — already done, just not written using it.** `feature_engine.creation.MathFeatures` still earns its place as a thin sklearn-`Pipeline`-compatible wrapper around this (so it fits into `fit`/`transform`/cross-validation), but it's not filling a pandas gap anymore — it's a convenience layer. The one piece truly missing from both is the per-value `count_i` tally (`(df[cols] == i).sum(axis=1)`), which is a one-liner, not something worth a feature request anywhere.

### `reduce_memory.py` (downcast dtypes) — pandas has taken *narrow* downcast PRs, never a general one

[pandas#33013](https://github.com/pandas-dev/pandas/issues/33013) (downcasting nullable dtypes in `to_numeric`) was accepted and merged (PR #38746) — evidence pandas *will* take scoped downcast improvements. But I found no open or closed issue asking for a general "shrink my whole DataFrame's dtypes automatically" feature in pandas core; the standalone tools that do this (`pandas-downcast`, `dtype-diet` — the latter by Ian Ozsvald, a well-known name in the Python data-tooling space) exist specifically *because* nobody has proposed this as core pandas scope, likely because pandas' current direction for memory efficiency is the structurally different Arrow-backed dtype system (`dtype_backend="pyarrow"`), not manual per-column int/float shrinking. **Big package target: pandas itself, via `dtype_backend="pyarrow"` — different mechanism, not a drop-in, and not what your script does.** No scikit-learn or statsmodels angle here at all; this is purely a pandas-memory-layout question.

### `better_pandas_describe.py` — no pandas issue found; this looks like a deliberate scope boundary, not an oversight

I could not find a pandas issue asking to extend `.describe()` with dtype/uniqueness/missing-rate/sentinel-value columns. That's consistent with pandas' actual behavior: `.describe()` has stayed narrow for years while an entire *separate* ecosystem of "one-call EDA summary" packages grew up around it — `ydata-profiling` (formerly `pandas-profiling`), `sweetviz`, `dtale`, `skimpy`, `klib`. That's a real pattern, not a coincidence: pandas keeps core methods minimal and treats richer reporting as an ecosystem concern. **No pandas/scikit-learn/statsmodels target** — the honest "big package" answer here is `ydata-profiling` (the biggest, most feature-complete member of that ecosystem, well beyond skimpy's scope) if you want maximal coverage, or `skimpy` if you want something closer in spirit to your original script.

### `IV_quantile.py` / `Stability_Indicator.py` (PSI) / `plot_feature_binning.py` — no issues found in any of the three; these are out-of-scope by design, not by neglect

I searched scikit-learn and statsmodels issue trackers for Information Value, Weight-of-Evidence binning, and Population Stability Index and found **nothing** — no open feature requests, no closed-as-wontfix, nothing. That absence is itself informative: these three techniques are specific to credit-risk/banking scorecard modeling (a regulated-industry practice, not a general statistical or ML technique), and both projects have a long-standing, explicit scope philosophy of shipping only well-established, broadly-applicable general-purpose methods — a domain-specific scorecard metric like IV or PSI simply never clears that bar, the same reason neither library ships, say, actuarial mortality tables. This isn't a gap anyone is waiting on; it's actively out of scope. **No pandas/scikit-learn/statsmodels target exists or is likely to.** The realistic "big package" answers stay domain-specific: `optbinning` (IV/binning), `evidently` (PSI/drift — corrected from `probatus`, which no longer has this at all, see above), `scorecardpy` (binning + plots) — all purpose-built for this niche, which is exactly why they exist as separate projects instead of PRs to the big three.

### `pandas_filter_on_read.py` — pandas discussed it, didn't build it; the real answer is a different big package entirely

[pandas#5900](https://github.com/pandas-dev/pandas/issues/5900) asked for chainable row filtering (`.filter(lambda row: ...)`) years ago and was closed without being built into pandas' I/O layer. Pandas' `read_csv` still has no predicate pushdown — it parses everything, then you filter. That gap didn't get filled by pandas; it got filled by a different big, actively-maintained package built around exactly this problem (lazy, columnar, pushdown-capable I/O): **`polars`**, via `scan_csv().filter(...)`. So the "big package target" here is real and actively maintained, it's just not one of the three named — worth widening the aperture past pandas/scikit-learn/statsmodels for this one specifically.

### `display_side_by_side.py` — not a library concern at all

No pandas issue found, and I'd argue there won't be one: rendering layout (putting two pieces of HTML next to each other) is a notebook-frontend/CSS concern, not something a dataframe library's job. **No pandas/scikit-learn/statsmodels target** — `itables` remains the right answer, but it's solving this by being a display library, not a pandas extension.

### `monitoring.sh` — not applicable

System memory monitoring has no connection to pandas, scikit-learn, or statsmodels; excluded from this comparison for that reason. `glances` (see maturity table above) is still the answer.

### Bottom line

Two real, useful surprises came out of checking the actual issue trackers instead of assuming: `AutoFE.py`'s core functionality is already native pandas (`df.agg([...], axis=1)`) — you can delete more of that file than I first said. And `pandas_filter_on_read.py`'s gap was considered and closed by pandas itself, then solved by `polars` instead — a genuine "look outside the big three" case. Everywhere else (IV/PSI/WOE-binning, the describe() extensions), the absence of any issue at all is the answer: these are deliberately out of scope for general-purpose libraries, not overlooked.

## Follow-up findings (verified, not assumed)

**Is there a fully-automatic "downcast everything" option?** No. Tested it directly: `df.convert_dtypes(dtype_backend="pyarrow")` does not shrink numeric column width at all — an `int64` column holding only values 0-99 stayed exactly 8000 bytes before and after, on a 1000-row test. Its real saving is on string/object columns: `object` dtype cost 58000 bytes for a 1000-row column of 3 repeated strings, `string[pyarrow]` cost 5000 bytes for the same data — over 11x smaller. So: no single call handles both cases automatically; you still need the explicit per-column int/float downcast loop (your script's approach) for numeric columns, and pyarrow-backed dtypes specifically for strings. The two are complementary, not substitutes for each other.

**`pandas_filter_on_read.py` — full working replacement, benchmarked:** verified on a 2,000,000-row synthetic CSV (~10% match rate on the filter). `polars.scan_csv(path).filter(pl.col("field") > constant).collect()` was 3.1x faster than the original chunked-pandas loop, and its output matched exactly (row-for-row, values equal). `pyarrow.dataset` with a `filter=` kwarg on `to_table()` also worked and matched, ~2.4x faster. One honest caveat: for a plain CSV (unindexed, row-by-row text), nothing can skip *reading* bytes on disk — pushdown here means the filter runs in a single fast native pass instead of pandas' per-chunk Python loop + `concat`, not that less data gets read. True skip-the-disk pushdown (e.g. skipping whole row groups) needs a columnar format like Parquet — worth converting to if this file gets re-read often.

**`monitoring.sh` → does `glances` actually do what yours does?** Verified by running it, not just reading docs — installed it and captured real output:

```
glances -t 60 --stdout-csv now,mem.total,mem.used,mem.free,mem.available,mem.percent,mem.shared,mem.buffers,mem.cached > log.csv
```

produces (confirmed live):

```
now.iso,now.custom,mem.total,mem.used,mem.free,mem.available,mem.percent,mem.shared,mem.buffers,mem.cached
2026-09-12T11:15:18+00:00,2026-09-12 11:15:18 UTC,8413380608,670834688,6895460352,7742545920,8.0,13619200,59863040,1050406912
```

Same fields as `free -m`'s Mem row, same 60-second interval, but real CSV with an ISO timestamp column instead of `sed`-scraped text. Annotated your original script with this command in `scripts/monitoring.sh`. One real gotcha found by testing: `-q`/`--quiet` silently suppresses `--stdout-csv` entirely (confirmed in its own log: "Quiet mode is ON, nothing will be displayed") - don't combine those two flags, a doc-only check would have missed this.

**`numpy-groupies` — candidate for numpy itself?** There's an open issue asking for exactly this: [numpy#7265](https://github.com/numpy/numpy/issues/7265), "Enhancement: groupby function," proposing `np.groupby(x, y).mean()`. It's been open since 2016 with no maintainer commitment either way — a decade of silence on a numpy core-team-relevant issue is itself the practical answer. This lines up with numpy's actual scope: arrays in numpy are unlabeled and homogeneous; "groupby" requires a labeled key array, which is precisely the concept pandas was built to add on top of numpy. Adding it to numpy core would blur that boundary rather than fill a gap - `numpy-groupies` (or pandas itself) staying separate is consistent with numpy's design, not an oversight. Didn't even need `numpy-groupies` in the end: pandas' own `.groupby(...).corr()` already reproduces `numpy_groupy_corr.py` exactly (see below) — no external package required.

**"Update my code with better alternatives as small comments, but they have to perform exactly" — verification pass results.** For every remaining script, I ran the proposed alternative against the original on the same data and compared actual numbers, not just APIs. Full annotated copies (original code + a verified comment block, same pattern as `monitoring.sh`) are in `annotated_originals/`. Summary of what was found, honest where "exactly" doesn't hold:

- **`reduce_memory.py` — a real bug, not just a better alternative.** On pandas 3.0.2 (the current `pip install pandas`), `col_type != 'object'` no longer catches plain string columns: pandas 3.0 turned on `future.infer_string` by default, so string columns now get dtype `str` (`pandas.StringDtype`), not `object`. Reproduced live: `pd.Series(np.array(['a','b','c'], dtype=object)).dtype` returns `'str'`. That sends string columns into the numeric branch of this function, which then crashes calling `.min()` on strings (`UFuncNoLoopError`). Fix is independent of any package choice: use `pandas.api.types.is_numeric_dtype()` / `is_string_dtype()` instead of comparing to the literal `'object'`. Separately, `pd.to_numeric(col, downcast=...)` reproduces the numeric-downcast loop with one real difference: it never downcasts to float16 (only float32), so on a `np.random.normal` test column your version picked float16 (smaller, ~3 significant digits) and `to_numeric` picked float32 (larger, full float precision) — not the same end memory number, and arguably your version is riskier there, not just different.

- **`Stability_Indicator.py` (PSI) — corrected recommendation.** `probatus` doesn't have PSI at all in its current release (see correction above) — `evidently`'s internal PSI function gives your exact `PSI_indicator(Base, New)` value (0.011262919669671231, bit-for-bit) once you divide evidently's result (which sums per-category terms, the standard convention) by the number of categories (your function takes a mean instead of a sum — a nonstandard normalization worth knowing either way).

- **`IV_quantile.py` and `plot_feature_binning.py` — "exactly" genuinely does not hold, and no configuration makes it hold.** Both use plain equal-frequency quantile bins. `optbinning.OptimalBinning` computes different numbers by design — it solves an actual bin-optimization problem (CP/MIP solver) rather than cutting at quantiles. Tested the closest possible case: handing `OptimalBinning` your exact 9 quantile cut points via `user_splits=` still returns a *different* IV (4.58 vs. your 6.39), because it merges bins to enforce monotonic WoE by default; forcing `user_splits_fixed=True` to stop that raises `ValueError: Fixed user_splits [...] are removed because produce pure prebins` outright — there's no supported way to make it a pass-through equal-frequency binner. Same finding applies to `plot_feature_binning.py`'s chart (same binning, just plotted). Speed check for `IV_quantile.py`: `OptimalBinning(solver="cp")` is ~9x slower than the quantile-only function on 100k rows (0.12s vs 0.013s) — real, but both are still sub-second.

- **`better_pandas_describe.py` — skimpy actually crashes on this function's own use case.** `skimpy.skim(df)` on a column containing `inf` (exactly what `values_list=[np.inf, -np.inf]` defaults to checking for) raises `ValueError: autodetected range of [1.0, inf] is not finite` from its histogram code. It also has no `constant_diff` row and no configurable sentinel-value counter. It's a fine `.describe()` upgrade in general, but not a working replacement for this specific function without pre-cleaning the data first.

- **`display_side_by_side.py` — itables solves a different problem, doesn't replace this.** `itables.show()` renders one interactive DataTable per call as its own block-level output; there's no argument for side-by-side inline layout, and it doesn't accept `df.style` objects. Keep this function if the layout is the point.

- **`AutoFE.py` and `pandas_filter_on_read.py`** — already verified in earlier passes (see the sections above); comments in `annotated_originals/` point at `autofe_native_pandas_equivalent.py` and `pandas_filter_on_read_alternatives.py` respectively, both of which include the numeric/output equality checks.

**`numpy_groupy_corr.py` — verdict corrected after actually benchmarking, not just checking correctness.** I previously said drop this in favor of `df.groupby('g')[cols].corr()` because the *values* match exactly (still true). But I hadn't benchmarked *speed* before recommending that, and it turns out to matter: benchmarked three regimes (500 groups x 400 rows, correctness cross-checked with `np.allclose` against pandas' native output every time):

| regime | numpy/joblib | pandas `.corr()` |
|---|---|---|
| 30 columns | 0.55s | 0.59s (even) |
| 100 columns | 1.75s | 2.57s (1.5x slower) |
| 5000 small groups, 30 cols | 1.68s | 1.60s (pandas slightly ahead) |

So the numpy version pulls ahead specifically as column count grows (pandas' `.corr()` seems to do per-column-pair work that scales worse than one `np.corrcoef` call per group) — filed as a performance issue draft (`issue_drafts/pandas_groupby_corr_perf.md`). It's a wash with few columns or many small groups. Also found: `Parallel(n_jobs=-1)` makes this *slower*, not faster, when groups are small and numerous (500 groups: 1.59s parallel vs. 0.55s serial) — multiprocessing dispatch overhead dominates. Kept the file, added `groupby_corr_by_index()` for calling it by column position instead of a DataFrame + column-name list.

## New module: `feature_interactions` (pairwise interaction screening)

Built from a `run_screen()` function you supplied, which loops over every feature pair, a small set of combining operators, and every target, quantile-bins the combined value, and compares how much variance that binning explains (eta-squared) versus each feature alone — a large gap between the pair's eta2 and the sum of the two solo eta2's flags a real interaction, not just two individually-useful features.

The skeleton called five functions that weren't included (`OPS`, `combine`, `base_eta2`, `rank_quantile`, `eta2_from_ranks`). Implemented at `src/dsutils/feature_interactions/screening.py`:

- `OPS = ["add", "sub", "mul", "div"]`, `combine(a, b, op)` — plain elementwise numpy; `div` produces inf on zero denominators by design, same as `run_screen`'s own `np.isfinite(combo)` mask already expects.
- `rank_quantile(x, n_bins)` — same equal-frequency quantile-binning convention already used in `IV_quantile.py` and `plot_feature_binning.py` (`np.unique(np.quantile(...))` then `np.digitize`), so behavior is consistent across all three. Returns `(None, None)` on a degenerate/near-constant input, matching those two scripts' own guard.
- `eta2_from_ranks(ranks, n_bins_actual, y)` — the standard one-way ANOVA correlation ratio, `SS_between / SS_total`. **Verified, not assumed:** cross-checked against `statsmodels.formula.api.ols(...)` + `sm.stats.anova_lm` on synthetic grouped data — exact match to floating-point tolerance (`tests/test_feature_interactions.py::test_eta2_matches_statsmodels_one_way_anova`).
- `base_eta2(...)` — per-target, per-column solo eta2, used as the "a alone"/"b alone" baseline.

**Does it actually catch a real interaction?** Built a synthetic case where `y = a*b + noise` and `a`, `b` are independently uniform (so each is individually almost uninformative about `y`, but their product is the whole signal) plus an unrelated noise column `c`. `run_screen` on this data: the `mul` combination of `a` and `b` has `combo_eta2 > 0.5` while `eta2_a` and `eta2_b` are both under `0.05`, and it's the single top-ranked row by `combo_eta2` across every pair/op/target combination in the output. That's the actual job this tool needs to do, and it does it (test: `test_run_screen_flags_a_real_multiplicative_interaction`).

10 tests total for this module, all passing, included in the 42/42 full suite.
