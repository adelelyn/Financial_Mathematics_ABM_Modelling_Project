from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional
from mesa import Agent
from scipy.stats import norm
from market import Order


def bs_call_price_delta(S: float, K: float, r: float, sigma: float, T: float):
    """
    Black–Scholes European call price + delta.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        price = max(S - K, 0.0)
        delta = 1.0 if S > K else 0.0
        return price, delta

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    return float(price), float(delta)


@dataclass
class Inventory:
    cash: float = 0.0
    stock: float = 0.0


class BaseTrader(Agent):
    def __init__(self, unique_id, model, inv: Optional[Inventory] = None):
        super().__init__(unique_id, model)
        self.inv = inv if inv is not None else Inventory()

    def place_limit(self, side: str, qty: int, price: float, tif: int = 5):
        self.model.lob.add_order(
            Order(
                self.unique_id, side, qty, price,
                self.model.t, tif,
                order_type="limit",
                is_market=False
            )
        )

    def place_market(self, side: str, qty: int):
        self.model.lob.add_order(
            Order(
                self.unique_id, side, qty, None,
                self.model.t, tif=1,
                order_type="market",
                is_market=True
            )
        )



class Fundamentalist(BaseTrader):
    def __init__(self, unique_id, model, fundamental_sigma: float = 0.02, aggression: float = 0.7):
        super().__init__(unique_id, model)
        self.fundamental_sigma = fundamental_sigma
        self.aggression = aggression

    def step(self):
        S_mid = self.model.current_mid()
        if S_mid is None:
            return
        F = self.model.fundamental_value
        mispricing = (F - S_mid) / S_mid

        qty = int(max(1, abs(mispricing) * 10))
        # If undervalued => buy; overvalued => sell
        if mispricing > 0:
            # place near best ask
            ba = self.model.lob.best_ask()
            px = ba if ba is not None else S_mid * (1 + 0.001)
            self.place_limit("buy", qty, px, tif=10)
        else:
            bb = self.model.lob.best_bid()
            px = bb if bb is not None else S_mid * (1 - 0.001)
            self.place_limit("sell", qty, px, tif=10)


class NoiseTrader(BaseTrader):
    def __init__(self, unique_id, model, p_trade: float = 0.25, max_qty: int = 3):
        super().__init__(unique_id, model)
        self.p_trade = p_trade
        self.max_qty = max_qty

    def step(self):
        if self.random.random() > self.p_trade:
            return
        side = "buy" if self.random.random() < 0.5 else "sell"
        qty = self.random.randint(1, self.max_qty)
        self.place_market(side, qty)


class HFT(BaseTrader):
    """
    Simplified HFT:
    - If short-term return signal is positive -> aggressively buy (take ask)
      else aggressively sell (take bid)
    - Optionally also posts tiny, short-lived quotes to capture spread.
    """
    def __init__(self, unique_id, model, p_take: float = 0.35, max_qty: int = 2, quote: bool = True):
        super().__init__(unique_id, model)
        self.p_take = p_take
        self.max_qty = max_qty
        self.quote = quote

    def step(self):
        S_mid = self.model.current_mid()
        if S_mid is None:
            return

        signal = self.model.short_signal()

        if self.random.random() < self.p_take:
            qty = self.random.randint(1, self.max_qty)
            if signal >= 0:
                self.place_market("buy", qty)
            else:
                self.place_market("sell", qty)

        if self.quote and self.random.random() < 0.5:
            spr = self.model.lob.spread()
            spr = spr if spr is not None else 0.02
            half = max(self.model.lob.tick, spr / 2)
            bid = S_mid - half
            ask = S_mid + half
            self.place_limit("buy", 1, bid, tif=1)
            self.place_limit("sell", 1, ask, tif=1)


class DeltaHedgingOptionMM(BaseTrader):
    """
    Market maker providing liquidity in STOCK, while managing option exposure:
    - Assumes it has a (exogenous) net option position q_opt (call).
    - Computes BS delta and targets stock inventory ~ -q_opt * delta
    - Posts quotes; if inventory deviates, skews quotes to mean-revert inventory.

    This is the minimal "delta-hedging options market maker" mechanism.
    """
    def __init__(
        self,
        unique_id,
        model,
        q_opt: float = 50.0,        
        risk_aversion: float = 0.02,
        base_spread: float = 0.04,
        max_quote_qty: int = 3
    ):
        super().__init__(unique_id, model)
        self.q_opt = q_opt
        self.risk_aversion = risk_aversion
        self.base_spread = base_spread
        self.max_quote_qty = max_quote_qty
        self.inventory = 0.0
        self.cash = 0.0

    def step(self):
        S_mid = self.model.current_mid()
        if S_mid is None:
            return

        sigma = self.model.sigma
        K, r, T = self.model.K, self.model.r, self.model.T
        _, delta = bs_call_price_delta(S_mid, K, r, sigma, T)

        target_stock = -self.q_opt * delta
        inv_err = self.inv.stock - target_stock

        spr = self.base_spread + self.risk_aversion * abs(inv_err) * 0.01
        skew = self.risk_aversion * inv_err * 0.01

        bid = S_mid - spr / 2 - skew
        ask = S_mid + spr / 2 - skew

        qty = int(max(1, min(self.max_quote_qty, 1 + abs(inv_err) // 10)))
        self.place_limit("buy", qty, bid, tif=5)
        self.place_limit("sell", qty, ask, tif=5)

        if abs(inv_err) > 15:
            hedge_qty = int(min(5, abs(inv_err)))
            if inv_err > 0:
                self.place_market("sell", hedge_qty)
            else:
                self.place_market("buy", hedge_qty)
