**Repo:** scikit-learn/scikit-learn
**Type:** Feature request

## Title
Add Information Value (IV) / Weight-of-Evidence binning as a feature-selection or preprocessing metric

## Describe the workflow you want to enable

Rank a feature's predictive strength against a binary target by binning it (equal-frequency or optimized) and computing Information Value: `sum((pct_event - pct_nonevent) * log(pct_event / pct_nonevent))` over bins. This is a standard univariate feature-screening step in binary classification, conceptually adjacent to things sklearn already ships in `sklearn.feature_selection` (`mutual_info_classif`, `f_classif`, chi2) as a way to rank/filter features before modeling.

## Describe your proposed solution

`sklearn.feature_selection.information_value(X, y, bins=10)` returning per-feature IV, using equal-frequency quantile binning by default (fast, matches the common textbook definition), with an optional `strategy='optimal'` for supervised bin optimization.

## Describe alternatives you've considered

Checked scikit-learn's issue tracker for "Information Value" and "Weight of Evidence" - found nothing (no open request, no closed-as-declined). The realistic existing option is `optbinning`, a purpose-built, actively maintained package that solves actual bin optimization (via a CP/MIP solver) rather than plain quantile cuts - verified it computes a genuinely different (not just differently-scaled) IV number than plain equal-frequency binning on the same data, since it's optimizing bin edges rather than reusing fixed quantile cuts. That's a real capability gap in favor of a dedicated package, not just a preference.

Per scikit-learn's own contributing FAQ ("at least 3 years since publication, 200+ citations, and wide use and usefulness" + general applicability), IV/WOE binning is old and widely used, but specifically within credit-scoring-style binary classification rather than general ML - similar concern to the adjacent PSI request I'm also filing. Raising it mainly to get a maintainer judgment on record, since I couldn't find this discussed anywhere in scikit-learn's history.

## Additional context

`mutual_info_classif` covers "how predictive is this feature," but it isn't a substitute for IV, and the reason is more concrete than "credit-risk practitioners are used to IV" - it's a real, verifiable mathematical difference, not a stylistic preference:

IV (`sum_i (p_i - q_i) * ln(p_i/q_i)`, with `p_i = P(bin=i|Y=1)`, `q_i = P(bin=i|Y=0)`) is exactly the Jeffreys (symmetric KL) divergence between the two class-conditional distributions - it compares class 1's distribution directly to class 0's, and never references the class prior `P(Y=1)` at all. Mutual information is the *prior-weighted* average KL divergence of each conditional to the pooled marginal `P(X) = P(Y=1)p + P(Y=0)q`. That weighting means MI is capped by `H(Y)`, so it shrinks toward 0 as the target gets rarer, regardless of how separable the feature actually is.

Checked this numerically, holding the feature's two conditional distributions fixed and only varying the class prior:

```
P(Y=1)=0.50: IV=2.2803  MI=0.2392 nats
P(Y=1)=0.10: IV=2.2803  MI=0.0948 nats
P(Y=1)=0.02: IV=2.2803  MI=0.0219 nats
```

(MI cross-checked against `sklearn.metrics.mutual_info_score` on 2M sampled rows from the same distributions - 0.0948 vs. 0.0954, matches within sampling noise.)

IV is completely unchanged as the event rate drops from 50% to 2%; MI collapses by over 10x on the exact same feature. Since credit default, fraud, and churn are all rare-event targets (typically 1-5%), `mutual_info_classif` would systematically undersell every feature's real separating power purely because the event is rare - not because the feature is weak. That's the actual reason IV persists as a distinct metric in that field rather than being subsumed by mutual information, not just documentation convention.
