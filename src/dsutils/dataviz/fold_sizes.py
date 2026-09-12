"""Horizontal stacked bar chart of dataset-split sizes per cross-validation fold.

Moved here from the standalone `plot_folds.py` script. Fixed two real bugs on
the way in:
- the original forced `plt.savefig(FOLDS_PATH + "folds_design.png")` as a side
  effect with no way to just get the Axes back (untestable, unusable if you
  wanted to show it inline or compose it into a larger figure);
- the original hardcoded `"Test": 1` for every fold, which is only correct if
  every fold happens to have exactly one test row.

No existing package match found for this (searched scikit-learn: it ships an
*example script*, `examples/model_selection/plot_cv_indices.py`, showing one
way to visualize CV behavior - not a public, importable plotting function.
No open issue asking for it to become one either. Low-value/niche either way;
kept only because you might already rely on it - not something worth
publishing as its own package.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.axes import Axes


def plot_fold_sizes(
    sizes: dict[str, list[float]] | pd.DataFrame,
    ax: Axes | None = None,
    **barh_kwargs,
) -> Axes:
    """Plot a stacked horizontal bar chart of split sizes per CV fold.

    Args:
        sizes: Maps a split name (e.g. "removed", "train", "val", "test") to
            one size per fold, in fold order. A dict-of-lists or a DataFrame
            both work; a dict is turned into a DataFrame internally.

            If you have the original tuple-per-fold format
            ``folds[f] = (removed, train_end, val_start, val_end)``, build
            this dict yourself, e.g.::

                sizes = {
                    "removed": [folds[f][0] for f in folds],
                    "train":   [folds[f][1] - folds[f][0] for f in folds],
                    "val":     [folds[f][3] - folds[f][2] for f in folds],
                    "test":    [test_size(f) for f in folds],  # your own logic
                }

        ax: Existing Axes to draw on. A new figure/axes is created if not given.
        **barh_kwargs: Forwarded to ``DataFrame.plot.barh`` (e.g. ``color={...}``,
            ``title=...``).

    Returns:
        The matplotlib Axes the chart was drawn on.
    """
    df = pd.DataFrame(sizes)
    if len(df) == 0:
        raise ValueError("sizes has no folds to plot")

    df = df.iloc[::-1]  # fold 0 at the top
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

    barh_kwargs.setdefault("stacked", True)
    df.plot.barh(ax=ax, **barh_kwargs)
    ax.set_ylabel("Fold")
    return ax
