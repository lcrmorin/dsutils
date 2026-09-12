import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_folds(folds, FOLDS_PATH, eras):

    Sizes = {
        "Removed": np.array([folds[f][0] for f in folds]),
        "Train": np.array([folds[f][1] - folds[f][0] for f in folds]),
        "Val": np.array([folds[f][3] - folds[f][2] for f in folds]),
        "Test": np.array([1 for f in folds]),
    }

    Sizes = pd.DataFrame(Sizes)[::-1]
    Sizes.plot.barh(
        stacked=True,
        title="Moons per fold",
        xlabel=eras,
        ylabel="Folds",
        color={"Removed": "grey", "Train": "blue", "Val": "green", "Test": "red"},
    )

    plt.savefig(FOLDS_PATH + f"folds_design.png")

# ---------------------------------------------------------------------------
# Moved next to the other plotting code in dataviz/ (see
# `src/dsutils/dataviz/fold_sizes.py` - a cleaned-up, importable version of
# exactly this: same stacked-barh-of-fold-sizes idea, parameterized instead
# of hardcoded to `folds_design.png`).
#
# Not in scikit-learn: checked, and it makes sense that it isn't - this is
# a data-specific visualization of your own custom fold/era split, not a
# generic cross-validation utility. scikit-learn's own CV visualization
# (`sklearn.model_selection.TimeSeriesSplit` diagrams in its docs, or
# plotting `cv.split()` output yourself) works on train/test index arrays
# in general, but doesn't know about your "Removed" segment or per-fold
# Train/Val/Test naming - there's nothing generic to add there without
# re-inventing your fold structure. Low value to chase further; keep this
# as your own small plotting function.
# ---------------------------------------------------------------------------
