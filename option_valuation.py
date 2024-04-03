import numpy as np
import math
import scipy.stats as stats
from functools import cache

def monte_carlo(S: float, K: float, r: float, v: float, T: float, j: int, trials: int, call: bool):
    """
    Monte Carlo Option Valuator for European Options using antithetic variance reduction
    @param S: Stock price
    @param r: Riskless rate annually (%)
    @param v: Volatility annually (%)
    @param T: Lifespan of option (non-inclusive), as proportion of a year
    @param j: Number of timesteps
    @param trials: Number of trials
    @param call: Whether option is a call or put
    @output mu_hat: Expected value of the option
    @output var_hat: Variance of the expected value
    @output s_paths: Paths of stock prices generated
    """
    assert isinstance(S, (float, int)) and isinstance(K, (float, int)) and isinstance(r, (float, int)) \
        and isinstance(v, (float, int)) and isinstance(T, (float, int)) and isinstance(j, int) \
        and isinstance(trials, int)
    
    #Create constants
    ln_S = math.log(S)
    dt = T/j
    rvt = (r - (0.5*(v**2))) * dt
    v_sqrt_dt = v * math.sqrt(dt)

    #Generate stock price paths
    epsilon = np.random.normal(size = (trials, j))
    epsilon = np.vstack((epsilon, -epsilon))
    ln_s_paths = ln_S + np.hstack((np.zeros((trials * 2, 1)), np.cumsum(rvt+(v_sqrt_dt*epsilon), axis = 1)))
    s_paths = np.exp(ln_s_paths)

    #Generate final option intrinsic values
    s_expiries = s_paths[:,-1]
    if call:
        v_expiries = np.where(s_expiries > K, s_expiries - K, 0)
    else:
        v_expiries = np.where(s_expiries < K, K - s_expiries, 0)
    mu_hat = np.mean(v_expiries)
    var_hat = np.std(v_expiries)

    return mu_hat, var_hat, s_paths

def lattice(S: float, K: float, r: float, v: float, T: float, j: int, call: bool, dividends = (), eu = False):
    """
    Lattice Option Valuator
    u and d are chosen to simulate GBM of stock price
    @param S: Stock price
    @param r: Riskless rate annually (%)
    @param v: Volatility annually (%)
    @param T: Lifespan of option (non-inclusive), as proportion of a year
    @param j: Number of timesteps
    @param call: Whether option is a call or put
    @output valuation
    """
    assert isinstance(S, (float, int)) and isinstance(K, (float, int)) and isinstance(r, (float, int)) \
        and isinstance(v, (float, int)) and isinstance(T, (float, int)) and isinstance(j, int)
    
    #Create constants
    dt = T/j
    ert = math.exp(-r*dt)
    u = math.exp(v * math.sqrt(dt))
    d = math.exp(- v * math.sqrt(dt))
    pi_ = (math.exp(r*dt) - d)/(u - d)

    #Function for stock value at lattice position
    def s_(t, du, dd):
        return (S*(u**du)*(d**dd)) + sum(-d_i * math.exp(r*(t-tau_i)) for d_i, tau_i in dividends if t > tau_i)

    @cache
    def lattice_valuation(du, dd, t_j):
        iv = max(s_(t_j, du, dd) - K, 0) if call else max(K - s_(t_j, du, dd), 0)
        if du+dd == j:
            return iv
        lattice_value = pi_ * lattice_valuation(du+1, dd, t_j+dt) + (1 - pi_) * lattice_valuation(du, dd+1, t_j+dt)
        return ert * lattice_value if eu else ert * max(lattice_value, iv)
    
    return lattice_valuation(0, 0, 0)