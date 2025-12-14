from __future__ import annotations
import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from market import SimpleLOB
from agents import Fundamentalist, NoiseTrader, HFT, DeltaHedgingOptionMM


class MarketABM(Model):
    """
    ABM market with:
    - Simple LOB
    - Fundamental process
    - Agents: Fundamentalists, Noise, HFT, Delta-hedging Option MM
    """

    def __init__(
        self,
        seed: int = 1,
        n_fund: int = 30,
        n_noise: int = 40,
        n_hft: int = 20,
        n_mm: int = 5,
        with_hft: bool = True,
        with_mm: bool = True,
        T_steps: int = 2000,
        S0: float = 100.0,
        fundamental_kappa: float = 0.01,
        fundamental_vol: float = 0.2,
        tick: float = 0.01,
        # option params for MM
        K: float = 100.0,
        r: float = 0.01,
        T: float = 30/365,
        sigma: float = 0.25
    ):
        super().__init__()
        self.random.seed(seed)
        np.random.seed(seed)

        self.t = 0
        self.T_steps = T_steps

        self.K, self.r, self.T, self.sigma = K, r, T, sigma

        self.lob = SimpleLOB(tick=tick)
        self.schedule = RandomActivation(self)

        self.fundamental_kappa = fundamental_kappa
        self.fundamental_vol = fundamental_vol
        self.fundamental_value = S0

        self.mid_series = []
        self.trade_series = []
        self.mm_inventory_series = []


        # Seed book with initial liquidity so mid exists
        self._seed_initial_book(S0)

        uid = 0
        for _ in range(n_fund):
            self.schedule.add(Fundamentalist(uid, self))
            uid += 1
        for _ in range(n_noise):
            self.schedule.add(NoiseTrader(uid, self))
            uid += 1

        if with_hft:
            for _ in range(n_hft):
                self.schedule.add(HFT(uid, self))
                uid += 1

        if with_mm:
            for _ in range(n_mm):
                self.schedule.add(DeltaHedgingOptionMM(uid, self))
                uid += 1

        self.datacollector = DataCollector(
            model_reporters={
                "mid": lambda m: m.current_mid(),
                "spread": lambda m: m.lob.spread(),
                "fundamental": lambda m: m.fundamental_value,
                "n_trades": lambda m: len(m.lob.trades),
                "last_trade": lambda m: m.lob.last_trade_price,
                "mm_inventory": lambda m: m.mm_inventory_series[-1]
            }
        )


    def _seed_initial_book(self, S0: float):
        self.lob.add_order(self._mk_order(agent_id=-1, side="buy", qty=10, price=S0 - 0.05, tif=10))
        self.lob.add_order(self._mk_order(agent_id=-2, side="sell", qty=10, price=S0 + 0.05, tif=10))

    def _mk_order(self, agent_id: int, side: str, qty: int, price: float, tif: int):
        from market import Order
        return Order(agent_id=agent_id, side=side, qty=qty, price=price, time=self.t, tif=tif)

    def current_mid(self):
        mid = self.lob.mid()
        if mid is None:
            return self.lob.last_trade_price if self.lob.last_trade_price is not None else self.fundamental_value
        return mid

    def short_signal(self, lookback: int = 5):
        if len(self.mid_series) < lookback + 1:
            return 0.0
        arr = np.array(self.mid_series[-(lookback+1):], dtype=float)
        rets = np.diff(np.log(arr))
        return float(np.mean(rets))

    def _update_fundamental(self):
        # mean-reverting fundamental
        # F_{t+1} = F_t + kappa*(S0 - F_t) + vol*eps
        eps = np.random.randn()
        drift = self.fundamental_kappa * (self.mid_series[0] - self.fundamental_value) if self.mid_series else 0.0
        shock = self.fundamental_vol * eps * 0.01
        self.fundamental_value = max(1e-6, self.fundamental_value * (1.0 + drift + shock))

    def step(self):
        self.lob.step_time(self.t)

        self._update_fundamental()
        self.schedule.step()

        self.mid_series.append(self.current_mid())
        if self.lob.last_trade_price is not None:
            self.trade_series.append(self.lob.last_trade_price)

        mm_inv = 0.0
        for agent in self.schedule.agents:
            if isinstance(agent, DeltaHedgingOptionMM):
                mm_inv += getattr(agent, "inventory", 0.0)
        self.mm_inventory_series.append(mm_inv)


        self.datacollector.collect(self)
        self.t += 1

    def run(self):
        for _ in range(self.T_steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()
