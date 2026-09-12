import numpy as np
import matplotlib.pyplot as plt

def plot_feature_binning(x, y, bins=10, feature_name="Feature"):
    mask = ~np.isnan(x)
    x = x[mask]
    y = y[mask]

    quantiles = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(quantiles) <= 2:
        print("Not enough unique bins to plot.")
        return

    bin_edges = quantiles[1:-1]
    x_binned = np.digitize(x, bin_edges, right=True)

    # Init arrays
    y0_counts = np.zeros(len(quantiles) - 1)
    y1_counts = np.zeros(len(quantiles) - 1)
    target_rates = np.zeros(len(quantiles) - 1)

    for b in range(len(quantiles) - 1):
        in_bin = x_binned == b
        bin_y = y[in_bin]
        y0_counts[b] = np.sum(bin_y == 0)
        y1_counts[b] = np.sum(bin_y == 1)
        target_rates[b] = np.mean(bin_y) if bin_y.size > 0 else np.nan

    idx = np.arange(len(y0_counts))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Bar plot on left y-axis
    ax1.bar(idx, y0_counts, color="skyblue", label="y = 0")
    ax1.bar(idx, y1_counts, bottom=y0_counts, color="salmon", label="y = 1")
    ax1.set_ylabel("Count")
    ax1.set_xlabel(feature_name)
    ax1.set_title(f"Binned Distribution of {feature_name} (Stacked + Target Rate)")

    # Twin y-axis for target rate
    ax2 = ax1.twinx()
    ax2.plot(idx, target_rates, color="black", marker="o", label="Target Rate")
    ax2.set_ylabel("Target Rate", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # X-axis bin labels
    bin_labels = [f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})" for i in range(len(quantiles) - 1)]
    ax1.set_xticks(idx)
    ax1.set_xticklabels(bin_labels, rotation=45)

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Same root cause as IV_quantile.py (this uses the identical plain
# equal-frequency quantile binning, just plotted instead of reduced to a
# single IV number), so the same finding applies here:
#
# optbinning's `BinningTable.plot(metric="event_rate")` draws the same
# *kind* of chart (bar counts + a target/event-rate line over bins), and
# takes a `metric="event_rate"` argument specifically so it plots the raw
# rate instead of its default WoE - but it will show DIFFERENT bins,
# because OptimalBinning re-optimizes bin edges rather than reusing plain
# quantile cuts (same `user_splits` test as in IV_quantile.py: handing it
# your exact 9 cut points still gets 8 back after its monotonicity pass).
# So "performs exactly" doesn't hold for the binning itself - only the
# general shape of what's being visualized (counts + rate per bin) is
# equivalent, not the bin edges or the resulting figure.
#
# If matching your exact quantile bins in the plot matters, keep this
# function; optbinning's plot is the right tool once you've accepted its
# (usually better-justified) bin choice instead of plain quantiles.
# ---------------------------------------------------------------------------
