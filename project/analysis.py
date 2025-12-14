import numpy as np
from market import realized_vol

def format_test_result(
    hypothesis: str,
    test_name: str,
    rationale: str,
    statistic: float,
    p_value: float,
    alpha: float = 0.05,
    extra: dict | None = None
):
    return {
        "hypothesis": hypothesis,
        "test": test_name,
        "rationale": rationale,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "reject_H0": p_value < alpha,
        **(extra or {})
    }


def price_inefficiency(df):
    return np.abs(df["mid"] - df["fundamental"])


def tail_events(returns, k=3):
    sigma = np.nanstd(returns)
    return np.abs(returns) > k * sigma


def compute_rv(df):
    mid = df["mid"].values
    rets = np.diff(np.log(mid))
    return realized_vol(rets)

def hft_aggression(trades, vol, q=0.75):
    thresh = np.quantile(vol, q)

    t = trades["t"].values
    mask = (t >= 0) & (t < len(vol))
    trades = trades.loc[mask].copy()
    t = t[mask]

    hi_mask = vol[t] > thresh
    lo_mask = ~hi_mask

    def frac_aggr(df):
        if len(df) == 0:
            return np.nan
        return np.mean(df["order_type"] == "market")

    aggr_hi = frac_aggr(trades.loc[hi_mask])
    aggr_lo = frac_aggr(trades.loc[lo_mask])

    return aggr_hi, aggr_lo

def implied_vol_proxy(mm_inventory):
    return np.abs(np.diff(mm_inventory))

def iv_rv_gap(iv, rv):
    return np.abs(iv - rv)
