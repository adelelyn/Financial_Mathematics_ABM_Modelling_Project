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
        
        price = self.model.current_price
        momentum = price - self.model.previous_price

        if momentum > 0:
            self.model.place_market_buy(self)
        else:
            self.model.place_market_sell(self)


class OptionsMarketMaker(Agent):

    def __init__(self, unique_id, model, option_position=0):
        super().__init__(unique_id, model)
        self.option_position = option_position
        self.stock_inventory = 0

    def compute_delta(self):
        S = self.model.current_price
        sigma = self.model.volatility
        T = self.model.T
        K = self.model.K
        r = self.model.r
        
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = 0.5 * (1 + np.math.erf(d1 / np.sqrt(2)))
        return delta

    def step(self):
        delta = self.compute_delta()
        target_stock = self.option_position * delta

        hedge_amount = target_stock - self.stock_inventory

        if hedge_amount > 0:
            self.model.place_market_buy(self, amount=abs(hedge_amount))
            self.stock_inventory += hedge_amount
        else:
            self.model.place_market_sell(self, amount=abs(hedge_amount))
            self.stock_inventory += hedge_amount
