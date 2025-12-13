from mesa import Agent
import numpy as np


class HighFrequencyTrader(Agent):
    def __init__(self, unique_id, model, speed=5):
        super().__init__(unique_id, model)
        self.speed = speed
        self.inventory = 0

    def step(self):
        if self.model.step_count % self.speed != 0:
            return

        if self.model.current_price > self.model.previous_price:
            self.model.place_market_buy(self)
        else:
            self.model.place_market_sell(self)


class OptionsMarketMaker(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.stock_inventory = 0
        self.option_position = 1

    def step(self):
        delta = 0.5
        target = self.option_position * delta
        hedge = target - self.stock_inventory

        if hedge > 0:
            self.model.place_market_buy(self, abs(hedge))
        else:
            self.model.place_market_sell(self, abs(hedge))

        self.stock_inventory += hedge
