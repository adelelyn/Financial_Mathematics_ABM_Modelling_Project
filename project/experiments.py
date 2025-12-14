import numpy as np
import pandas as pd
from model import MarketABM
import statsmodels.api as sm
from scipy.stats import spearmanr, ttest_ind, f_oneway, chi2_contingency
from analysis import format_test_result, price_inefficiency, tail_events, compute_rv

import os

def save_results(results, path="results/hypothesis_tests.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    return df


SCENARIOS = {
    "no_hft_no_mm": dict(with_hft=False, with_mm=False),
    "hft_only":     dict(with_hft=True,  with_mm=False),
    "hft_mm":       dict(with_hft=True,  with_mm=True),
}


def H1a_test(dfs_hft, dfs_no_hft):
    A = np.concatenate([price_inefficiency(df) for df in dfs_hft])
    B = np.concatenate([price_inefficiency(df) for df in dfs_no_hft])

    stat, p = ttest_ind(A, B, equal_var=False)

    # one-sided: HFT < no-HFT
    p = p / 2 if np.mean(A) < np.mean(B) else 1 - p / 2

    return format_test_result(
        hypothesis="H1a: Price discovery improves with HFT",
        test_name="t-test",
        rationale="Comparison of mean price inefficiency; CLT justifies t-test with unequal variances",
        statistic=stat,
        p_value=p,
        extra={
            "mean_HFT": np.mean(A),
            "mean_no_HFT": np.mean(B)
        }
    )

def H1b_test(groups):
    data = {k: [compute_rv(df) for df in v] for k, v in groups.items()}

    stat, p = f_oneway(*data.values())

    return format_test_result(
        hypothesis="H1b: Volatility increases with HFT and options",
        test_name="ANOVA",
        rationale="Comparison of mean realized volatility across multiple market configurations",
        statistic=stat,
        p_value=p,
        extra={f"mean_{k}": np.mean(v) for k, v in data.items()}
    )



def H1c_test(dfs_hft_mm, dfs_no_hft):
    def count_extreme(dfs):
        cnt = 0
        tot = 0
        for df in dfs:
            r = np.diff(np.log(df["mid"]))
            flags = tail_events(r)
            cnt += flags.sum()
            tot += len(flags)
        return cnt, tot - cnt

    table = np.array([
        count_extreme(dfs_hft_mm),
        count_extreme(dfs_no_hft)
    ])

    stat, p, _, _ = chi2_contingency(table)

    return format_test_result(
        hypothesis="H1c: Tail risk increases with HFT and options",
        test_name="Chi-squared",
        rationale="Extreme events are binary outcomes; test compares frequencies",
        statistic=stat,
        p_value=p
    )


def H2a_test(aggr_hi, aggr_lo):
    stat, p = ttest_ind(aggr_hi, aggr_lo, equal_var=False)

    p = p / 2 if np.mean(aggr_hi) > np.mean(aggr_lo) else 1 - p / 2

    return format_test_result(
        hypothesis="H2a: HFT aggression increases in high volatility regimes",
        test_name="t-test",
        rationale="Comparison of mean aggressive order share across volatility regimes",
        statistic=stat,
        p_value=p,
        extra={
            "mean_high_vol": np.mean(aggr_hi),
            "mean_low_vol": np.mean(aggr_lo)
        }
    )


def H2b_test(rets, aggression):
    X = sm.add_constant(aggression)
    y = np.abs(rets)
    model = sm.OLS(y, X).fit()

    beta = model.params[1]
    p = model.pvalues[1]

    return format_test_result(
        hypothesis="H2b: Aggressive HFT amplifies short-term price movements",
        test_name="OLS (t-test on β)",
        rationale="Linear regression tests marginal effect of aggression on volatility",
        statistic=beta,
        p_value=p
    )

def H3a_test(gap_stress, gap_normal):
    stat, p = ttest_ind(gap_stress, gap_normal, equal_var=False)

    return format_test_result(
        hypothesis="H3a: IV–RV gap increases in stress regimes",
        test_name="Welch t-test",
        rationale="Comparison of mean IV–RV gap across regimes with unequal variance",
        statistic=stat,
        p_value=p
    )

def H3b_test(gap, activity):
    rho, p = spearmanr(gap, activity)

    return format_test_result(
        hypothesis="H3b: IV–RV gap correlates with agent activity",
        test_name="Spearman correlation",
        rationale="Monotonic relationship without assuming linearity",
        statistic=rho,
        p_value=p
    )


def run_scenario(cfg, seed, T=2000):
    m = MarketABM(seed=seed, T_steps=T, **cfg)
    df = m.run()
    trades = pd.DataFrame(m.lob.trades)
    return df, trades


def run_all_scenarios(
    scenarios,
    seeds,
    T=2000
):
    """
    Returns:
    data[scenario] = {
        "dfs":    list[pd.DataFrame],
        "trades": list[pd.DataFrame]
    }
    """
    out = {}

    for name, cfg in scenarios.items():
        dfs, trades = [], []

        for seed in seeds:
            df, tr = run_scenario(cfg, seed, T=T)
            dfs.append(df)
            trades.append(tr)

        out[name] = {
            "dfs": dfs,
            "trades": trades
        }

    return out
