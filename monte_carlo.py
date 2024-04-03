import numpy as np
import scipy.stats as stats
from random_variable import RandomVariable
import plotly

class MonteCarlo():
    def __init__(self, rvs: tuple[RandomVariable], foo):
        """
        Constructs a Monte Carlo estimator object for a given function of random variables
        @param rvs: tuple of RandomVariable objects to be used in the function - must be listed in
            order of evaluation - i.e. if one random variable depends on another - the dependent
            variable must appear later in the tuple
        @param foo: the function to be performed on the random variables - the order of arguments
            should be the same as the order of the random variables passed in rvs
        """
        self.rvs = rvs
        self.foo = foo

    def e(self, trials):
        """
        Generates an expected value, variance of the expected value, and 95% confidence interval
        using Monte Carlo estimation with 'trials' trials
        @param trials: number of trials
        @output: expected value, variance of the expected value, and 95% confidence interval
        """
        values = np.array([self.foo(*(rv.evaluate() for rv in self.rvs)) for _ in range(trials)])
        mu_bar = np.mean(values)
        var_bar = np.std(values)
        confidence_interval = [mu_bar-(1.96*(np.sqrt(var_bar/trials))), mu_bar+(1.96*(np.sqrt(var_bar/trials)))]
        return mu_bar, var_bar, confidence_interval
