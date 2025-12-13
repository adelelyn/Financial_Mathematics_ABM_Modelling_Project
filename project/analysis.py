import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp
from market import realized_vol


def compute_metrics(df: pd.DataFrame) -> dict:
    mid = df["mid"].astype(float).values
    spread = df["spread"].astype(float).values

    # log returns
    rets = np.diff(np.log(mid))
    rv = realized_vol(rets, annualize_factor=252.0)

    out = {
        "realized_vol": rv,
        "avg_spread": np.nanmean(spread),
        "spread_p95": np.nanpercentile(spread[~np.isnan(spread)], 95) if np.any(~np.isnan(spread)) else np.nan,
        "kurtosis_proxy": np.nanmean(((rets - np.nanmean(rets)) / (np.nanstd(rets) + 1e-12))**4) if len(rets) > 10 else np.nan,
        "n_obs": len(mid)
    }
    return out


def run_tests(metric_list_A, metric_list_B, key: str, alpha: float = 0.05):
    A = np.array([m[key] for m in metric_list_A], dtype=float)
    B = np.array([m[key] for m in metric_list_B], dtype=float)

    A = A[~np.isnan(A)]
    B = B[~np.isnan(B)]

    res = {}

    # t-test
    t_stat, t_p = ttest_ind(A, B, equal_var=False)
    res["t_test"] = {"stat": float(t_stat), "p": float(t_p), "reject": bool(t_p < alpha)}

    # Mann-Whitney
    if len(A) > 0 and len(B) > 0:
        u_stat, u_p = mannwhitneyu(A, B, alternative="two-sided")
        res["mann_whitney"] = {"stat": float(u_stat), "p": float(u_p), "reject": bool(u_p < alpha)}

    return res


def ks_test_returns(dfA: pd.DataFrame, dfB: pd.DataFrame, alpha: float = 0.05):
    midA = dfA["mid"].astype(float).values
    midB = dfB["mid"].astype(float).values
    rA = np.diff(np.log(midA))
    rB = np.diff(np.log(midB))
    stat, p = ks_2samp(rA, rB)
    return {"ks_stat": float(stat), "p": float(p), "reject": bool(p < alpha)}
