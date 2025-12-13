from mesa import Model
from mesa.time import RandomActivation
import numpy as np

from agents import HighFrequencyTrader, OptionsMarketMaker
from market import MarketEnvironment

class MarketModel(Model):
    def __init__(self, N_hft=10, N_mm=2):
        self.schedule = RandomActivation(self)
        self.market = MarketEnvironment()

        self.current_price = self.market.current_price
        self.previous_price = self.market.previous_price

        self.step_count = 0

        for i in range(N_hft):
            hft = HighFrequencyTrader(i, self)
            self.schedule.add(hft)

        for j in range(N_hft, N_hft + N_mm):
            mm = OptionsMarketMaker(j, self)
            self.schedule.add(mm)

        self.sigma = 0.2
        self.T = 1
        self.K = 100
        self.r = 0.01

    def place_market_buy(self, agent, amount=1):
        self.market.place_market_buy(agent, amount)
        self.current_price = self.market.current_price

    def place_market_sell(self, agent, amount=1):
        self.market.place_market_sell(agent, amount)
        self.current_price = self.market.current_price

    def step(self):
        self.step_count += 1
        self.schedule.step()
