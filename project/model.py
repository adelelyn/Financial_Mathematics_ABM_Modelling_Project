from mesa import Model
from mesa.time import RandomActivation

from agents import HighFrequencyTrader, OptionsMarketMaker
from market import MarketEnvironment


class MarketModel(Model):
    def __init__(self, N_hft=5, N_mm=1):
        self.schedule = RandomActivation(self)
        self.market = MarketEnvironment()

        self.current_price = self.market.current_price
        self.previous_price = self.market.previous_price

        self.step_count = 0

        for i in range(N_hft):
            self.schedule.add(HighFrequencyTrader(i, self))

        for j in range(N_hft, N_hft + N_mm):
            self.schedule.add(OptionsMarketMaker(j, self))

    def step(self):
        self.step_count += 1
        self.schedule.step()
        self.current_price = self.market.current_price
