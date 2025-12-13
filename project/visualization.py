import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_series(df: pd.DataFrame, title: str = "Market"):
    fig = plt.figure()
    plt.plot(df["mid"].values, label="mid")
    plt.plot(df["fundamental"].values, label="fundamental", alpha=0.7)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_spread(df: pd.DataFrame, title: str = "Spread"):
    fig = plt.figure()
    plt.plot(df["spread"].values)
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_return_hist(df: pd.DataFrame, title: str = "Log returns"):
    mid = df["mid"].astype(float).values
    rets = np.diff(np.log(mid))
    fig = plt.figure()
    plt.hist(rets, bins=60)
    plt.title(title)
    plt.tight_layout()
    return fig
