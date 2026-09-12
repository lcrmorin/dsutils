import matplotlib

matplotlib.use("Agg")

import pytest

from dsutils.dataviz.fold_sizes import plot_fold_sizes


def test_basic_stacked_bars():
    sizes = {
        "removed": [10, 5, 0],
        "train": [100, 120, 140],
        "val": [20, 20, 20],
        "test": [1, 1, 1],
    }
    ax = plot_fold_sizes(sizes)
    # 3 folds * 4 stacked series = 12 bar patches
    assert len(ax.patches) == 12


def test_variable_test_size_not_hardcoded_to_one():
    """Regression test: the original script hardcoded Test=1 for every fold."""
    sizes = {"train": [100, 100], "test": [5, 50]}
    ax = plot_fold_sizes(sizes)
    heights_or_widths = [p.get_width() for p in ax.patches]
    assert 50 in [round(w) for w in heights_or_widths]


def test_empty_sizes_raises():
    with pytest.raises(ValueError):
        plot_fold_sizes({"train": []})


def test_accepts_dataframe_directly():
    import pandas as pd

    df = pd.DataFrame({"train": [10, 20], "val": [1, 2]})
    ax = plot_fold_sizes(df)
    assert len(ax.patches) == 4
