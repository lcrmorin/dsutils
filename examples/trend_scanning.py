"""Reproducible trend-scanning demo. Run: python examples/trend_scanning.py."""

import matplotlib.pyplot as plt
import numpy as np

from dsutils.time_series import scan_trends


def main():
    rng = np.random.default_rng(42)
    x = np.arange(72)
    values = np.where(x > 36, 1, -1).cumsum() + 20 * (rng.random(len(x)) - .5)
    scan = scan_trends(values, min_lookback=6)
    best = scan.loc[scan.t_stat.abs().idxmax()]
    lookback = int(best.name)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    axes[0].plot(x, values, color="0.3", label="Synthetic observations")
    for length, fit in scan.iterrows():
        axes[0].plot(x[-length:], fit.intercept + fit.slope * np.arange(length),
                     color="steelblue", alpha=.08)
    axes[0].plot(x[-lookback:], best.intercept + best.slope * np.arange(lookback),
                 color="crimson", linewidth=2, label=f"Largest |t|: {lookback} observations")
    axes[0].set(xlabel="Observation", ylabel="Value", title="Trailing trend scan")
    axes[0].legend()
    axes[1].plot(scan.index, scan.t_stat, color="steelblue")
    axes[1].axhline(0, color="0.6", linewidth=.8)
    axes[1].set(xlabel="Lookback (observations)", ylabel="Signed slope t-statistic")
    print(scan.loc[[lookback]].to_string())
    fig.savefig("trend_scanning.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
