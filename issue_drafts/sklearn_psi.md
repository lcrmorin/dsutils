**Repo:** scikit-learn/scikit-learn
**Type:** Feature request

## Title
Add Population Stability Index (PSI) as a model/data drift metric

## Describe the workflow you want to enable

Detect when a feature's or a model score's distribution has shifted between two datasets (e.g. training vs. a more recent production sample), using Population Stability Index - `sum((new_pct - base_pct) * log(new_pct / base_pct))` over aligned bins/categories. This is one of the most common "has my model gone stale" checks in production ML monitoring, sitting next to `sklearn.metrics` conceptually (it's a distance/divergence metric between two empirical distributions, similar in spirit to things like `scipy.stats.entropy`/KL divergence that sklearn already depends on via scipy).

## Describe your proposed solution

`sklearn.metrics.population_stability_index(base, new, bins=10)` for continuous input (auto-binning via quantiles of `base`) and direct category alignment for categorical/discrete input, returning the scalar PSI (and optionally a per-bin breakdown table).

## Describe alternatives you've considered

Checked scikit-learn's and statsmodels' issue trackers directly for prior discussion of PSI - found none in either (no open request, nothing closed as wontfix). Purpose-built alternatives exist as separate packages (I found and verified `evidently`'s internal PSI implementation matches a reference PSI computation exactly), which is consistent with PSI being viewed as a monitoring-pipeline concern rather than a general ML metric - flagging this issue mainly to get that judgment on record from a maintainer, given I couldn't find it discussed anywhere already. Understand this is likely to run into scikit-learn's inclusion bar from the contributing FAQ ("well-established... at least 3 years since publication, 200+ citations, and wide use") - PSI is old and widely used operationally, but almost entirely within credit-risk/banking-adjacent modeling rather than general ML, so it may not clear the "broadly applicable" bar even though it clears "well-established."

## Additional context

Population Stability Index has been standard practice in credit scoring model monitoring since well before modern ML tooling (commonly attributed to the SAS/banking scorecard literature from the 1990s-2000s); today it's also used more broadly for feature/data drift detection outside credit scoring specifically. Happy to be pointed to why this is intentionally out of scope if that's the answer - couldn't find that documented anywhere for this specific metric.
