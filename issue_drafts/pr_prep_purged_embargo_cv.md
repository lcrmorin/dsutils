**This is a PR-prep note, not an issue to open now.** scikit-learn#22229 ("Combinatorial Purged Cross-Validation strategy") and scikit-lego#541 ("PurgedGroupTimeSeriesSplit") already asked for this - both closed, neither implemented, no maintainer response on either. Filing a third issue just repeats an ask that's already sat unanswered twice; the better move is to finish your own implementation and open a PR directly, linking both stale issues as the prior asks it resolves (a working implementation is a much stronger prompt for maintainer attention than another request). Use the template below for that PR description once your splitter is ready - `sklearn.model_selection` PRs are expected to include tests and a docstring with a runnable example, so bring those along with the code.

---

**Repo:** scikit-learn/scikit-learn
**Type:** Pull request (not an issue)

## Title
ENH: Add `PurgedTimeSeriesSplit` (embargo + purge gaps for time-series CV)

## What does this implement/fix?

Adds a `PurgedTimeSeriesSplit(n_splits, embargo, purge)` cross-validator alongside `TimeSeriesSplit` in `sklearn.model_selection`. Standard `TimeSeriesSplit` orders folds correctly but doesn't remove training samples whose label-computation window overlaps the validation fold ("purging"), or add a buffer after validation before training resumes ("embargo") - both standard practice for time-ordered/autocorrelated-label data (documented in the purged/embargoed CV literature), and both currently missing from scikit-learn and from any actively-maintained extension library.

## Prior discussion

- scikit-learn#22229 - asked for this (as "Combinatorial Purged CV"), closed with no response.
- scikit-lego#541 - asked for a narrower version (`PurgedGroupTimeSeriesSplit`), closed, never implemented.

Only small, low-adoption personal packages fill this gap today (`timeseriescv`, `eslazarev/purged-cross-validation`) - nothing with scikit-learn's reach or maintenance guarantee.

## Testing done

<fill in once implemented - the load-bearing test is a leakage check: assert no training index falls within `embargo` distance of any validation index, and no training index's label window overlaps the validation fold under `purge`>

## Additional context

Happy to iterate on API naming/shape based on maintainer feedback - opening as a working PR rather than a fresh feature request given the history above.
