import numpy as np
import scipy.stats as stats

class RandomVariable:
    def __init__(self, distribution_function, distribution_args: tuple):
        """
        Construct a random variable object from a distribution and variables
        @param distribution_function: scipy.stats distribution to be used to sample
        @param distribution_args: tuple of arguments for the distribution '.rvs' function 
            - can include RandomVariable objects
        """
        self.foo = distribution_function
        self.args = distribution_args

    def evaluate(self):
        """
        Generate a sample from the random variable
        """
        self.value = self.foo.rvs(*(arg.value if isinstance(arg, RandomVariable) else arg for arg in self.args))
        return self.value