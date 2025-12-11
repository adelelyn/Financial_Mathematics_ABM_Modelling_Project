from mesa import Agent
import numpy as np

class HFTrader(Agent):
    def __init__(self, unique_id, model, speed = 5):
        super().__init__(unique_id, model)
        self.speed = speed
        self.inventory = 0

#class OptionsMarketMaker(Agent):

