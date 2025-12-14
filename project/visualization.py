import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_price_vs_fundamental(
    df: pd.DataFrame,
    title: str = "Price vs Fundamental"
):
    fig, ax = plt.subplots()
    ax.plot(df["mid"].values, label="Mid price")
    ax.plot(df["fundamental"].values, label="Fundamental", alpha=0.7)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_price_inefficiency_dist(
    ineff_a: np.ndarray,
    ineff_b: np.ndarray,
    labels=("Scenario A", "Scenario B"),
    title="Price inefficiency |P − F|"
):
    fig, ax = plt.subplots()
    ax.hist(ineff_a, bins=50, density=True, alpha=0.6, label=labels[0])
    ax.hist(ineff_b, bins=50, density=True, alpha=0.6, label=labels[1])
    ax.set_xlabel("|P − F|")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_return_histogram(
    df: pd.DataFrame,
    title: str = "Log-return distribution"
):
    mid = df["mid"].astype(float).values
    rets = np.diff(np.log(mid))

    fig, ax = plt.subplots()
    ax.hist(rets, bins=60, density=True)
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_volatility_comparison(
    rv_a: np.ndarray,
    rv_b: np.ndarray,
    labels=("Scenario A", "Scenario B"),
    title="Realized volatility comparison"
):
    fig, ax = plt.subplots()
    ax.hist(rv_a, bins=30, density=True, alpha=0.6, label=labels[0])
    ax.hist(rv_b, bins=30, density=True, alpha=0.6, label=labels[1])
    ax.set_xlabel("Realized volatility")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_tail_ccdf(
    df: pd.DataFrame,
    title: str = "Tail risk (CCDF)"
):
    mid = df["mid"].astype(float).values
    rets = np.abs(np.diff(np.log(mid)))

    x = np.sort(rets)
    y = 1.0 - np.arange(1, len(x) + 1) / len(x)

    fig, ax = plt.subplots()
    ax.loglog(x, y)
    ax.set_xlabel("|log return|")
    ax.set_ylabel("P(|r| > x)")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_aggressiveness_vs_volatility(
    volatility: np.ndarray,
    aggressive_share: np.ndarray,
    title: str = "HFT aggressiveness vs volatility"
):
    fig, ax = plt.subplots()
    ax.scatter(volatility, aggressive_share, alpha=0.5)
    ax.set_xlabel("Realized volatility")
    ax.set_ylabel("Aggressive order share")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_price_impact_comparison(
    abs_ret_aggr: np.ndarray,
    abs_ret_base: np.ndarray,
    labels=("After aggressive HFT", "Baseline"),
    title="Price impact of aggressive trading"
):
    fig, ax = plt.subplots()
    ax.hist(abs_ret_aggr, bins=50, density=True, alpha=0.6, label=labels[0])
    ax.hist(abs_ret_base, bins=50, density=True, alpha=0.6, label=labels[1])
    ax.set_xlabel("|log return|")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_iv_rv_gap_timeseries(
    gap: np.ndarray,
    title: str = "IV–RV gap over time"
):
    fig, ax = plt.subplots()
    ax.plot(gap)
    ax.set_xlabel("Time")
    ax.set_ylabel("|IV − RV|")
    ax.set_title(title)
    fig.tight_layout()
    return fig

def plot_iv_rv_gap_distribution(
    gap_stress: np.ndarray,
    gap_normal: np.ndarray,
    labels=("Stress", "Normal"),
    title="IV–RV gap: stress vs normal"
):
    fig, ax = plt.subplots()
    ax.hist(gap_stress, bins=40, density=True, alpha=0.6, label=labels[0])
    ax.hist(gap_normal, bins=40, density=True, alpha=0.6, label=labels[1])
    ax.set_xlabel("|IV − RV|")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def plot_gap_vs_activity(
    activity: np.ndarray,
    gap: np.ndarray,
    title: str = "IV–RV gap vs agent activity"
):
    fig, ax = plt.subplots()
    ax.scatter(activity, gap, alpha=0.5)
    ax.set_xlabel("Agent activity (hedging / volume)")
    ax.set_ylabel("|IV − RV|")
    ax.set_title(title)
    fig.tight_layout()
    return fig
