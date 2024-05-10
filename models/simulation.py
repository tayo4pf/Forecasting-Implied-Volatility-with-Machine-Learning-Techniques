import numpy as np
import datetime

class Simulation:
    def __init__(self, simulations, period, scale, start):
        self.simulations = simulations[:,:,:period]
        self.length = len(simulations)
        self.period = period
        self.start = start
        self.end = self.start + datetime.timedelta(days=self.period)
        self.i = 0
        self.scale = scale

    def __getitem__(self, index):
        return self.simulations[index, :, :self.period] * self.scale

    def __iter__(self):
        self.i = 0
        return self
    
    def __next__(self):
        if self.i >= len(self.simulations):
            self.i = 0
            raise StopIteration
        self.i += 1
        return self[self.i-1]