import numpy as np

class MarketEnvironment:

    def __init__(self, initial_price=100):
        self.current_price = initial_price
        self.previous_price = initial_price
        self.order_book = []

    def place_market_buy(self, agent, amount=1):
        self.previous_price = self.current_price
        self.current_price += 0.1 * amount

    def place_market_sell(self, agent, amount=1):
        self.previous_price = self.current_price
        self.current_price -= 0.1 * amount
