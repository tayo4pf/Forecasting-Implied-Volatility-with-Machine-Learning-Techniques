import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import distributions
import datetime
import arch
import pmdarima as pm
from pmdarima.model_selection import RollingForecastCV, cross_val_predict
import pandas as pd
from simulation import Simulation
import py_vollib.black_scholes as black_scholes_merton
import py_vollib_vectorized

def call_option_ev(simulations, s, k):
        """
        Function to find the expected value of an option using Monte Carlo estimation
        simulations: an nparray, indexed by timestep within the expiration period, then simulation
        @param s: current price of the underlying
        @param k: strike price of the option
        """
        value_over_period = ((s * np.exp(simulations.sum(axis=1))) - k)
        value_over_period[value_over_period < 0] = 0
        ev = np.mean(value_over_period)
        return ev

def call_option_dist(simulations, s, k):
    """
    Function to find the distribution of an option value
    simulations: an nparray, indexed by timestep within the expiration period, then simulation
    @param s: current price of the underlying
    @param k: strike price of the option
    @return: np.array of values
    """
    value_over_period = ((s * np.exp(simulations.sum(axis=1))) - k)
    value_over_period[value_over_period < 0] = 0
    return value_over_period

def put_option_ev(simulations, s, k):
    """
    Function to find the expected value of an option using Monte Carlo estimation
    simulations: an nparray, indexed by timestep within the expiration period, then simulation
    @param s: current price of the underlying
    @param k: strike price of the option
    """
    value_over_period = (k - (s * np.exp(simulations.sum(axis=1))))
    value_over_period[value_over_period < 0] = 0
    ev = np.mean(value_over_period)
    return ev

def put_option_dist(simulations, s, k):
    """
    Function to find the distribution of an option value
    simulations: an nparray, indexed by timestep within the expiration period, then simulation
    @param s: current price of the underlying
    @param k: strike price of the option
    """
    value_over_period = (k - (s * np.exp(simulations.sum(axis=1))))
    value_over_period[value_over_period < 0] = 0
    return value_over_period

class ARCHModel:
    def __init__(self, data:pd.DataFrame, split_date:datetime, model:str, p:int, q:int, volatility_period:int):
        self.underlying_data = data
        self.split_date = split_date

        self.underlying_data["Realized Volatility"] = distributions.get_realized_volatility(
            self.underlying_data,
            volatility_period
        )

        #Training model
        model = arch.arch_model(
            self.underlying_data["Residuals"] * 100, 
            mean="Zero",
            p=p,
            q=q,
            vol=model
        )
        self.res = model.fit(last_obs=self.split_date)
        self.forecast = None
        self.forecast_df = None
        self.forecast_residuals = None
        self.forecast_i = 0

    def forecaster(self, period:int):
        self.forecast = self.res.forecast(
            horizon=period, start=self.split_date, method="simulation", simulations=1000
            )
        self.forecast_df = pd.DataFrame(
            {
            "Forecasted Period Volatility":(self.forecast.variance.mean(axis=1) * np.sqrt(252)/100)[:-21],
            },
            index = self.forecast.variance.index[period:]
        )
        self.forecast_residuals = self.forecast_df["Forecasted Period Volatility"] - self.underlying_data["Realized Volatility"]

    def simulation(self, period):
        if self.forecast is None:
            raise NotImplementedError("You must first make a forecast")
        if period > self.forecast.simulations.values.shape[2]:
            raise ValueError("Period cannot be greater than forecast period")
        return Simulation(self.forecast.simulations.values, period, 1/100)

class ARCHForecastModel(ARCHModel):
    def __init__(self, ticker:str, start:datetime, end:datetime, split_date:datetime, model:str, p:int, q:int, volatility_period:int):
        #Storing values
        self.ticker = ticker
        self.start = start
        self.split_date = split_date
        self.end = end
        self.p = p
        self.q = q
        
        #Collecting data
        self.underlying_returns = distributions.get_returns(ticker, start, end)
        self.underlying_data = pd.DataFrame({"Log Returns":self.underlying_returns})
        self.underlying_data["Cumulative Returns"] = np.exp(self.underlying_data["Log Returns"].cumsum())
        self.underlying_data = self.underlying_data.dropna(axis=0, how="all")
        self.underlying_data["Adj Close"] = yf.download(ticker, start=start, end=end)["Adj Close"]
        self.underlying_data.index = pd.to_datetime(self.underlying_data.index)
        self.underlying_data["Realized Volatility"] = distributions.get_realized_volatility(
            self.underlying_data["Log Returns"],
            volatility_period
        )
        
        #Training model
        model = arch.arch_model(
            self.underlying_data["Log Returns"] * 100, 
            mean="Constant",
            p=p,
            q=q,
            vol=model
        )
        self.res = model.fit(last_obs=split_date)
        self.forecast = None
        self.forecast_df = None
        self.forecast_residuals = None
        self.forecast_i = 0

    def options_pricing(self, simulator, moneyness, shift=True):
        """
        Returns Dataframe with call, and put, forecasted pricing
        using monte carlo estimation with the passed simulator object
        @param simulator: Simulator object to be used for estimation
        @param moneyness: fraction into the money the option is
        @param shift: whether to shift the date to the expiry date (True) or leave it at date
        of forecast (optional)
        @return: Dataframe
        """
        #Creating dataframe with pricing
        pricing = pd.DataFrame({
            "Call Price ATM": (call_option_ev(sim_, price, price*moneyness)\
                                for sim_, price in zip(simulator, self.underlying_data["Adj Close"][self.split_date:])),
            "Realized Call Value ATM": (self.underlying_data["Adj Close"][self.split_date:].shift(-simulator.period) -\
                                        self.underlying_data["Adj Close"][self.split_date:]),
            "Put Price ATM": (put_option_ev(sim_, price, price*moneyness)\
                                for sim_, price in zip(simulator, self.underlying_data["Adj Close"][self.split_date:])),
            "Realized Put Value ATM": (-self.underlying_data["Adj Close"][self.split_date:].shift(-simulator.period) +\
                                        self.underlying_data["Adj Close"][self.split_date:]),
            "Underlying Price": self.underlying_data["Adj Close"][self.split_date:]},
            index = self.forecast.variance.index)
        #Fixing realized values to not contain negative values
        pricing["Realized Call Value ATM"][pricing["Realized Call Value ATM"] < 0] = 0
        pricing["Realized Put Value ATM"][pricing["Realized Put Value ATM"] < 0] = 0
        #Shifting to expiry date if shift is True
        if shift:
            pricing = pricing.shift(simulator.period)

        pricing = pricing.dropna(axis=0, subset="Call Price ATM")
        
        return pricing

    def plot_conditional_vol_rolling(self, period:int):
        plt.plot(self.res.conditional_volatility.rolling(window=period).mean()*np.sqrt(252)/100)

class ARIMAForecastModel:
    def __init__(self,ticker:str, start:datetime, end:datetime, split_date:datetime, volatility_period:int, order:tuple, seasonal_order:tuple=(0,0,0,0)):
        """
        Initializes an ARIMA Forecast Model Object
        @param ticker: The ticker of the underlying to be modelled
        @param start: The start date of the training data used in the model
        @param end: The end date of the data to be modelled
        @param split_data: The end date of the training data, and the start date of the data to be modelled
        @param order: The ARIMA order to be used
        @param seasonal_order: The ARIMA seasonal order to be used
        @param volatility_period: The time period used to calculate the volatility
        """
        #Storing values
        self.ticker = ticker
        self.start = start
        self.split_date = split_date
        self.end = end
        self.order = order
        self.seasonal_order = seasonal_order
        
        #Collecting data
        self.underlying_returns = distributions.get_returns(ticker, start, end)
        self.underlying_data = pd.DataFrame({"Log Returns":self.underlying_returns})
        self.underlying_data["Cumulative Returns"] = np.exp(self.underlying_data["Log Returns"].cumsum())
        self.underlying_data = self.underlying_data.dropna(axis=0, how="all")
        self.underlying_data["Adj Close"] = yf.download(ticker, start=start, end=end)["Adj Close"]
        self.underlying_data.index = pd.to_datetime(self.underlying_data.index)
        self.underlying_data["Realized Volatility"] = distributions.get_realized_volatility(
            self.underlying_data["Log Returns"],
            volatility_period
        )
        
        #Training model
        self.model = pm.arima.ARIMA(
            order=order,
            seasonal_order=seasonal_order
        )
        self.res = self.model.fit(self.underlying_returns[:split_date]*100)
        self.forecast = None
        self.forecast_df = None
        self.forecast_residuals = None
        self.forecast_i = 0

    def forecaster(self, period:int):
        """
        Produces an n day forecast within the model, where n is the "period"
        @param period: Number of days ahead to forecast
        """
        #Creates rolling forecast cross-validator to add up-to-date information
        #to forecasts past the split date
        cv = RollingForecastCV(h=period, step=1, initial=len(self.underlying_data["Log Returns"][:self.split_date]))
        self.forecast = cross_val_predict(
            self.model, 
            y=self.underlying_data["Log Returns"],
            cv=cv, return_raw_predictions=True
            )
        forecasted_volatility = np.std(self.forecast/100, axis=1, ddof=0)*np.sqrt(252)
        self.forecast_df = pd.DataFrame(
            {
                "Forecasted Period Volatility": forecasted_volatility
            }, index=self.underlying_data.index
        )
        self.forecast_df = self.forecast_df.shift(period)
        self.forecast_residuals = self.forecast_df["Forecasted Period Volatility"] - self.underlying_data["Realized Volatility"]

    def simulation(self, period):
        """
        Creates a simulation object containing simulated paths of the returns from the model
        @param period: Number of days to be simulated in each simulation
        @return: Simulation object
        """
        cv = RollingForecastCV(h=period, step=1, initial=len(self.underlying_data["Log Returns"][:self.split_date]))
        data_generator = cv.split(self.underlying_data["Log Returns"])
        sim = np.array([self.model.fit(self.underlying_data["Log Returns"][indices[0]]).arima_res_.simulate(nsimulations=period, repetitions=1000).T/100 for indices in data_generator])
        return Simulation(sim, period, 1/100)
    
    def options_pricing(self, simulator, moneyness, shift=True):
        """
        Returns Dataframe with call, and put, forecasted pricing
        using monte carlo estimation with the passed simulator object
        @param simulator: Simulator object to be used for estimation
        @param moneyness: fraction into the money the option is
        @param shift: whether to shift the date to the expiry date (True) or leave it at date
        of forecast (optional)
        @return: Dataframe
        """
        #Creating dataframe with pricing
        pricing = pd.DataFrame({
            "Call Price ATM": (call_option_ev(sim_, price, price*moneyness)\
                                for sim_, price in zip(simulator, self.underlying_data["Adj Close"][self.split_date:])),
            "Realized Call Value ATM": (self.underlying_data["Adj Close"][self.split_date:].shift(-simulator.period) -\
                                        self.underlying_data["Adj Close"][self.split_date:]),
            "Put Price ATM": (put_option_ev(sim_, price, price*moneyness)\
                                for sim_, price in zip(simulator, self.underlying_data["Adj Close"][self.split_date:])),
            "Realized Put Value ATM": (-self.underlying_data["Adj Close"][self.split_date:].shift(-simulator.period) +\
                                        self.underlying_data["Adj Close"][self.split_date:]),
            "Underlying Price": self.underlying_data["Adj Close"][self.split_date:]},
            index = self.underlying_data["Adj Close"][self.split_date:][:-simulator.period+1].index)
        #Fixing realized values to not contain negative values
        pricing["Realized Call Value ATM"][pricing["Realized Call Value ATM"] < 0] = 0
        pricing["Realized Put Value ATM"][pricing["Realized Put Value ATM"] < 0] = 0
        #Shifting to expiry date if shift is True
        if shift:
            pricing = pricing.shift(simulator.period)

        pricing = pricing.dropna(axis=0, subset="Call Price ATM")
        
        return pricing

class ARIMAARCHForecastModel:
    def __init__(self, ticker:str, start:datetime, end:datetime, split_date:datetime, volatility_period:int, p:int, q:int, model:str, order:tuple, seasonal_order:tuple=(0,0,0,0)):
        """
        Initializes an ARIMA Forecast Model Object
        @param ticker: The ticker of the underlying to be modelled
        @param start: The start date of the training data used in the model
        @param end: The end date of the data to be modelled
        @param split_data: The end date of the training data, and the start date of the data to be modelled
        @param order: The ARIMA order to be used
        @param seasonal_order: The ARIMA seasonal order to be used
        @param volatility_period: The time period used to calculate the volatility
        """
        #Storing values
        self.ticker = ticker
        self.start = start
        self.split_date = split_date
        self.end = end
        self.volatility_period = volatility_period
        self.order = order
        self.seasonal_order = seasonal_order
        self.p = p
        self.q = q
        self.model = model
        
        #Collecting data
        self.underlying_returns = distributions.get_returns(ticker, start, end)
        self.underlying_data = pd.DataFrame({"Log Returns":self.underlying_returns})
        self.underlying_data["Cumulative Returns"] = np.exp(self.underlying_data["Log Returns"].cumsum())
        self.underlying_data = self.underlying_data.dropna(axis=0, how="all")
        self.underlying_data["Adj Close"] = yf.download(ticker, start=start, end=end)["Adj Close"]
        self.underlying_data.index = pd.to_datetime(self.underlying_data.index)
        self.underlying_data["Realized Volatility"] = distributions.get_realized_volatility(
            self.underlying_data["Log Returns"],
            volatility_period
        )
        
        #Training ARIMA and GARCH Models
        self.arima_model = ARIMAForecastModel(ticker, start, end, split_date, volatility_period, order, seasonal_order)
        self.arch_model = ARCHModel(pd.DataFrame({"Residuals":(self.arima_model.res.predict_in_sample()/100) - self.underlying_data["Log Returns"]}, index=self.underlying_data.index).replace(np.nan, 0), split_date, self.model, self.p, self.q, self.volatility_period)
        self.forecast = None
        self.forecast_df = None
        self.forecast_residuals = None
        self.forecast_i = 0

    def forecaster(self, period:int):
        self.arima_model.forecaster(period)
        self.arch_model.forecaster(period)

    def simulation(self, period:int):
        """
        Creates a simulation object containing simulated paths of the returns from the model
        @param period: Number of days to be simulated in each simulation
        @return: Simulation object
        """
        arima_sim = self.arima_model.simulation(period)
        arch_sim = self.arch_model.simulation(period)
        sim = (arima_sim.simulations*arima_sim.scale) + (arch_sim.simulations[:-period+1,:,:]*arch_sim.scale)
        return Simulation(sim, period, 1)
    
    def options_pricing(self, simulator, moneyness, shift=True):
        """
        Returns Dataframe with call, and put, forecasted pricing
        using monte carlo estimation with the passed simulator object
        @param simulator: Simulator object to be used for estimation
        @param moneyness: fraction into the money the option is
        @param shift: whether to shift the date to the expiry date (True) or leave it at date
        of forecast (optional)
        @return: Dataframe
        """
        #Creating dataframe with pricing
        pricing = pd.DataFrame({
            "Call Price ATM": (call_option_ev(sim_, price, price*moneyness)\
                                for sim_, price in zip(simulator, self.underlying_data["Adj Close"][self.split_date:])),
            "Realized Call Value ATM": (self.underlying_data["Adj Close"][self.split_date:].shift(-simulator.period) -\
                                        self.underlying_data["Adj Close"][self.split_date:]),
            "Put Price ATM": (put_option_ev(sim_, price, price*moneyness)\
                                for sim_, price in zip(simulator, self.underlying_data["Adj Close"][self.split_date:])),
            "Realized Put Value ATM": (-self.underlying_data["Adj Close"][self.split_date:].shift(-simulator.period) +\
                                        self.underlying_data["Adj Close"][self.split_date:]),
            "Underlying Price": self.underlying_data["Adj Close"][self.split_date:]},
            index = self.underlying_data["Adj Close"][self.split_date:][:-simulator.period+1].index)
        #Fixing realized values to not contain negative values
        pricing["Realized Call Value ATM"][pricing["Realized Call Value ATM"] < 0] = 0
        pricing["Realized Put Value ATM"][pricing["Realized Put Value ATM"] < 0] = 0
        #Shifting to expiry date if shift is True
        if shift:
            pricing = pricing.shift(simulator.period)

        pricing = pricing.dropna(axis=0, subset="Call Price ATM")
        
        return pricing

class HistoricVolModel:
    def __init__(self, ticker:str, start:datetime, end:datetime, split_date:datetime, volatility_period:int, historic_r_period:int, historic_vol_period:int):
        """
        Initializes an ARIMA Forecast Model Object
        @param ticker: The ticker of the underlying to be modelled
        @param start: The start date of the training data used in the model
        @param end: The end date of the data to be modelled
        @param split_data: The end date of the training data, and the start date of the data to be modelled
        @param volatility_period: The time period used to calculate the volatility
        """
        #Storing values
        self.ticker = ticker
        self.start = start
        self.split_date = split_date
        self.end = end
        self.volatility_period = volatility_period
        self.historic_r_period = historic_r_period
        self.historic_vol_period = historic_vol_period

        #Collecting data
        self.underlying_returns = distributions.get_returns(ticker, start, end)
        self.underlying_data = pd.DataFrame({"Log Returns":self.underlying_returns})
        self.underlying_data["Cumulative Returns"] = np.exp(self.underlying_data["Log Returns"].cumsum())
        self.underlying_data = self.underlying_data.dropna(axis=0, how="all")
        self.underlying_data["Adj Close"] = yf.download(ticker, start=start, end=end)["Adj Close"]
        self.underlying_data.index = pd.to_datetime(self.underlying_data.index)
        self.underlying_data["Realized Volatility"] = distributions.get_realized_volatility(
            self.underlying_data["Log Returns"],
            volatility_period
        )

    def forecaster(self, period:int):
        pass

    def simulation(self, period:int):
        """
        Creates a simulation object containing simulated paths of the returns from the model
        @param period: Number of days to be simulated in each simulation
        @return: Simulation object
        """
        historic_r = self.underlying_data["Log Returns"].rolling(window=self.historic_r_period).mean()
        historic_vol = self.underlying_data["Log Returns"].rolling(window=self.historic_vol_period).std(ddof=0)

        sim = np.array([[
            np.random.normal(mu,var,size=period)
            for _ in range(1000)
        ] for mu, var in zip(historic_r[self.split_date:],historic_vol[self.split_date:])])
        return Simulation(sim, period, 1)
    
    def options_pricing(self, simulator, moneyness, shift=True):
        """
        Returns Dataframe with call, and put, forecasted pricing
        using monte carlo estimation with the passed simulator object
        @param simulator: Simulator object to be used for estimation
        @param moneyness: fraction into the money the option is
        @param shift: whether to shift the date to the expiry date (True) or leave it at date
        of forecast (optional)
        @return: Dataframe
        """
        #Creating dataframe with pricing
        pricing = pd.DataFrame({
            "Call Price ATM": (call_option_ev(sim_, price, price*moneyness)\
                                for sim_, price in zip(simulator, self.underlying_data["Adj Close"][self.split_date:])),
            "Realized Call Value ATM": (self.underlying_data["Adj Close"][self.split_date:].shift(-simulator.period) -\
                                        self.underlying_data["Adj Close"][self.split_date:]),
            "Put Price ATM": (put_option_ev(sim_, price, price*moneyness)\
                                for sim_, price in zip(simulator, self.underlying_data["Adj Close"][self.split_date:])),
            "Realized Put Value ATM": (-self.underlying_data["Adj Close"][self.split_date:].shift(-simulator.period) +\
                                        self.underlying_data["Adj Close"][self.split_date:]),
            "Underlying Price": self.underlying_data["Adj Close"][self.split_date:]},
            index = self.underlying_data["Adj Close"][self.split_date:].index)
        #Fixing realized values to not contain negative values
        pricing["Realized Call Value ATM"][pricing["Realized Call Value ATM"] < 0] = 0
        pricing["Realized Put Value ATM"][pricing["Realized Put Value ATM"] < 0] = 0
        #Shifting to expiry date if shift is True
        if shift:
            pricing = pricing.shift(simulator.period)

        pricing = pricing.dropna(axis=0, subset="Call Price ATM")
        
        return pricing