from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np


@dataclass
class Order:
    agent_id: int
    side: str          # "buy" or "sell"
    qty: int
    price: Optional[float]  # None => market order
    time: int
    tif: int = 1       # time-in-force (steps); for fast-cancel / short-lived limits


class SimpleLOB:
    """
    Very simple price-time priority LOB:
    - Keeps two lists: bids (desc), asks (asc)
    - Matches marketable orders immediately
    - Allows short-lived limit orders via TIF expiration
    """

    def __init__(self, tick: float = 0.01):
        self.tick = float(tick)
        self.bids: List[Order] = []
        self.asks: List[Order] = []
        self.last_trade_price: Optional[float] = None
        self.trades: List[Dict] = []
        self._t: int = 0

    def step_time(self, t: int):
        self._t = t
        self._expire_orders()

    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return 0.5 * (bb + ba)

    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return None
        return ba - bb

    def add_order(self, order: Order):
        if order.qty <= 0:
            return
        if order.price is not None:
            order.price = self._round_to_tick(order.price)

        # Try match if marketable
        if order.side == "buy":
            self._match_buy(order)
            if order.qty > 0 and order.price is not None:
                self._insert_limit(self.bids, order, reverse=True)
        else:
            self._match_sell(order)
            if order.qty > 0 and order.price is not None:
                self._insert_limit(self.asks, order, reverse=False)

    def _round_to_tick(self, p: float) -> float:
        return round(p / self.tick) * self.tick

    def _insert_limit(self, book: List[Order], order: Order, reverse: bool):
        # price-time priority
        book.append(order)
        if reverse:  # bids descending
            book.sort(key=lambda o: (-o.price, o.time))
        else:        # asks ascending
            book.sort(key=lambda o: (o.price, o.time))

    def _match_buy(self, order: Order):
        while order.qty > 0 and self.asks:
            best = self.asks[0]
            # If limit buy, only match if price >= best ask
            if order.price is not None and order.price < best.price:
                break
            trade_qty = min(order.qty, best.qty)
            trade_price = best.price
            order.qty -= trade_qty
            best.qty -= trade_qty
            self.last_trade_price = trade_price
            self.trades.append({
                "t": self._t, "price": trade_price, "qty": trade_qty,
                "buyer": order.agent_id, "seller": best.agent_id
            })
            if best.qty == 0:
                self.asks.pop(0)

    def _match_sell(self, order: Order):
        while order.qty > 0 and self.bids:
            best = self.bids[0]
            # If limit sell, only match if price <= best bid
            if order.price is not None and order.price > best.price:
                break
            trade_qty = min(order.qty, best.qty)
            trade_price = best.price
            order.qty -= trade_qty
            best.qty -= trade_qty
            self.last_trade_price = trade_price
            self.trades.append({
                "t": self._t, "price": trade_price, "qty": trade_qty,
                "buyer": best.agent_id, "seller": order.agent_id
            })
            if best.qty == 0:
                self.bids.pop(0)

    def _expire_orders(self):
        # Remove orders whose tif ended
        def alive(o: Order) -> bool:
            return (self._t - o.time) < o.tif

        self.bids = [o for o in self.bids if alive(o) and o.qty > 0]
        self.asks = [o for o in self.asks if alive(o) and o.qty > 0]


def realized_vol(returns: np.ndarray, annualize_factor: float = 252.0) -> float:
    if len(returns) < 2:
        return np.nan
    return np.nanstd(returns, ddof=1) * np.sqrt(annualize_factor)
