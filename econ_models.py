from typing import Union
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
"""
Module for econometric option pricing and IV forecasting models
"""

def call_option_ev(simulations: np.array, s: float, k: float, r: float = 1.0) -> float:
        """
        Function to find the expected undiscounted value of an option using Monte Carlo estimation
        @param simulations: an nparray, indexed by timestep within the expiration period, then simulation
        @param s: current price of the underlying
        @param k: strike price of the option
        """
        dte = simulations.shape[1]
        value_over_period = ((s * np.exp(simulations.sum(axis=1))) - k)/(r**(dte/252))
        value_over_period[value_over_period < 0] = 0
        ev = np.mean(value_over_period)
        return ev

def call_option_dist(simulations: np.array, s: float, k: float, r: float = 1.0) -> np.array:
    """
    Function to find the distribution of an undiscounted option value
    @param simulations: an nparray, indexed by timestep within the expiration period, then simulation
    @param s: current price of the underlying
    @param k: strike price of the option
    @return: np.array of values
    """
    dte = simulations.shape[1]
    value_over_period = ((s * np.exp(simulations.sum(axis=1))) - k)/(r**(dte/252))
    value_over_period[value_over_period < 0] = 0
    return value_over_period

def put_option_ev(simulations: np.array, s: float, k: float, r: float = 1.0) -> float:
    """
    Function to find the expected undiscounted value of an option using Monte Carlo estimation
    @param simulations: an nparray, indexed by timestep within the expiration period, then simulation
    @param s: current price of the underlying
    @param k: strike price of the option
    """
    dte = simulations.shape[1]
    value_over_period = (-(s * np.exp(simulations.sum(axis=1))) + k)/(r**(dte/252))
    value_over_period[value_over_period < 0] = 0
    ev = np.mean(value_over_period)
    return ev

def put_option_dist(simulations: np.array, s: float, k: float, r: float = 1.0) -> np.array:
    """
    Function to find the distribution of an undiscounted option value
    @param simulations: an nparray, indexed by timestep within the expiration period, then simulation
    @param s: current price of the underlying
    @param k: strike price of the option
    """
    dte = simulations.shape[1]
    value_over_period = (-(s * np.exp(simulations.sum(axis=1))) + k)/(r**(dte/252))
    value_over_period[value_over_period < 0] = 0
    return value_over_period

def iv_surface_dataframe(model, moneynesses: np.array, maturities: np.array) -> pd.core.frame.DataFrame:
    """
    Returns dataframe with option pricing and implied volatility data for a given list of moneynesses and maturities
    If the model is an ARCH Forecast Model, then a forecast for the maximum maturity must be run before this function is called
    @param model: a forecast model object
    @param moneynesses: list of moneynesses to generate data for
    @param maturities: list of maturities to generate data for  
    """
    # Creating dataframe
    ivs_df = pd.DataFrame()

    # Iterating maturities
    for dte in maturities:
        # Generating simulations for lifetime of the option
        simulator = model.simulation(dte)

        # Iterating moneynesses
        for moneyness in moneynesses:
            options = model.options_pricing(simulator, moneyness, shift=False)
            options = options.reset_index()
            options["Moneyness"] = moneyness
            options["DTE"] = dte
            ivs_df = pd.concat((ivs_df, options), ignore_index=True)
    
    return ivs_df

def plot_iv_surface(ivs_df: pd.core.frame.DataFrame, spot_date: Union[str, datetime.datetime]):
    """
    Plots IV Surface for a given implied volatility surface dataframe (iv_surface_dataframe)
    @param model: Forecast model
    @param moneynesses: Array-like of moneynesses in the IV surface
    @param maturities: Array-like of maturities (DTE) in the IV surface
    @param spot_date: Spot date of surface to be displayed
    """
    # Calls
    spot_chain = ivs_df[ivs_df["Date"] == spot_date].dropna()

    dte_fig = px.line_3d(spot_chain, x="DTE", y="Moneyness", z="Call IV", color="DTE")
    lm_fig = px.line_3d(spot_chain, x="DTE", y="Moneyness", z="Call IV", color="Moneyness")
    fig = go.Figure(data=dte_fig.data + lm_fig.data)

    fig.update_layout(title=f'Forecasted Call IV Surface for spot date: {spot_date}')
    fig.update_scenes(xaxis_title_text="Days Till Expiry",
                    yaxis_title_text="Moneyness",  
                    zaxis_title_text="Call IV")

    fig.show(renderer="notebook")

    spot_chain = ivs_df[ivs_df["Date"] == spot_date].dropna()

    dte_fig = px.line_3d(spot_chain, x="DTE", y="Moneyness", z="Put IV", color="DTE")
    lm_fig = px.line_3d(spot_chain, x="DTE", y="Moneyness", z="Put IV", color="Moneyness")
    fig = go.Figure(data=dte_fig.data + lm_fig.data)

    fig.update_layout(title=f'Forecasted Put IV Surface for Spot Date {spot_date}')
    fig.update_scenes(xaxis_title_text="Days Till Expiry",  
                    yaxis_title_text="Moneyness",  
                    zaxis_title_text="Put IV")

    fig.show(renderer="notebook")

class EconForecastModel:
    def forecaster(self):
        pass

    def simulation(self):
        pass

    def option_pricing(self):
        pass

class ARCHModel:
    def __init__(self, data:pd.DataFrame, split_date:datetime, volatility_period:int, model:str, p:int, q:int, dist:str = "normal", mean:str = "Zero"):
        self.underlying_data = data
        self.split_date = split_date

        self.underlying_data["Realized Volatility"] = distributions.get_realized_volatility(
            self.underlying_data,
            volatility_period
        )

        #Training model
        model = arch.arch_model(
            self.underlying_data["Residuals"] * 100, 
            mean=mean,
            p=p,
            q=q,
            vol=model,
            dist=dist
        )
        self.res = model.fit(last_obs=self.split_date)
        self.forecast = None
        self.forecast_df = None
        self.forecast_residuals = None
        self.forecast_i = 0

    def forecaster(self, period:int):
        # Generate forecast
        self.forecast = self.res.forecast(
            horizon=period, start=self.split_date, method="simulation", simulations=1000
            )
        
        # Define forecast dataframe
        self.forecast_df = pd.DataFrame(
            {
            "Forecasted Period Volatility":(np.sqrt(self.forecast.variance).mean(axis=1) * np.sqrt(252)/100)[:-21],
            },
            index = self.forecast.variance.index[period:]
        )
        self.forecast_residuals = self.forecast_df["Forecasted Period Volatility"] - self.underlying_data["Realized Volatility"]

    def simulation(self, period):
        # Check forecast has been generated
        if self.forecast is None:
            raise NotImplementedError("You must first make a forecast")
        if period > self.forecast.simulations.values.shape[2]:
            raise ValueError("Period cannot be greater than forecast period")
        return Simulation(self.forecast.simulations.values, period, 1/100, self.split_date)

class ARCHForecastModel(ARCHModel, EconForecastModel):
    """
    Forecaster for option price and implied volatility based on modelling returns with ARCH
    """
    def __init__(self, ticker:str, start:datetime, end:datetime, split_date:datetime, volatility_period:int, model:str, p:int, q:int, dist:str = "normal"):
        """
        Initializes an ARCH Forecast Model Object
        @param ticker: Ticker of the underlying security to be forecasted
        @param start: The start date for the training data for returns to be modelled with
        @param end: The end date for the model to forecast returns on
        @param split_date: The date at which training data ends and the forecast begins
        @param model: Model used for volatility process (GARCH, EGARCH)
        @param p: ARCH model p value
        @param q: ARCH model q value
        @param volatility_period: Period for realized volatility to be measured over
        """
        # If dates provided are strings, convert them to datetimes
        if isinstance(start, str):
            start = datetime.datetime.strptime(start,"%Y-%m-%d")
        if isinstance(split_date, str):
            split_date = datetime.datetime.strptime(split_date,"%Y-%m-%d")
        if isinstance(end, str):
            end = datetime.datetime.strptime(end, "%Y-%m-%d")

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
        self.dividends = yf.Ticker(ticker).dividends/100
        self.dividends.index = self.dividends.index.tz_convert(None)
        self.rates = distributions.get_risk_free_rate(None, None)
        
        #Training model
        _model = arch.arch_model(
            self.underlying_data["Log Returns"] * 100, 
            mean="Constant",
            p=p,
            q=q,
            vol=model,
            dist=dist
        )
        self.res = _model.fit(last_obs=split_date)
        self.forecast = None
        self.forecast_df = None
        self.forecast_residuals = None
        self.forecast_i = 0

    def options_pricing(self, simulator, moneyness, shift=True):
        """
        Returns Dataframe with discounted call, and put, forecasted pricing
        using monte carlo estimation with the passed simulator object
        @param simulator: Simulator object to be used for estimation
        @param moneyness: fraction into the money the option is (S/K)
        @param shift: whether to shift the date to the expiry date (True) or leave it at date
        of forecast (optional)
        @return: Dataframe
        """
        #Creating dataframe with pricing
        pricing = pd.DataFrame({
            "Underlying Price": self.underlying_data["Adj Close"][self.split_date:]},
            index = self.forecast.variance.index)

        # Creating Expiry date column
        pricing["Expiry Date"] = pricing.index + datetime.timedelta(days=simulator.period)

        # Applying discount
        pricing = pricing.join(self.rates, how="left")

        # Joining Dividend Yields
        dividends = self.dividends.reindex(
            index = pd.date_range(self.dividends.index[0], self.dividends.index[-1]),
            method = "ffill"
        )
        dividends.index = dividends.index.date
        pricing = pricing.join(dividends, how="left").rename(columns={"Dividends":"Dividend Yield"})

        # Finding realized values and fixing realized values to not contain negative values
        # We fill because options expire across weekends, not only trading days
        data = yf.download(self.ticker, self.start, self.end + datetime.timedelta(days=simulator.period))
        filled_underlying_data = data["Adj Close"].reindex(
            index = pd.date_range(self.start, self.end + datetime.timedelta(days=simulator.period)),
            method = "ffill"
        )
        pricing = pricing.join(filled_underlying_data.shift(-simulator.period), how="left").rename(columns={"Adj Close":"Expired Underlying Price"}) 
        pricing["Strike Price"] = pricing["Underlying Price"]/moneyness
        pricing["Realized Call Value"] = (pricing["Expired Underlying Price"] - pricing["Strike Price"])/(pricing["Daily Rate"]**simulator.period)
        pricing["Realized Put Value"] = (-pricing["Expired Underlying Price"] + pricing["Strike Price"])/(pricing["Daily Rate"]**simulator.period)
        pricing["Realized Call Value"][pricing["Realized Call Value"] < 0] = 0
        pricing["Realized Put Value"][pricing["Realized Put Value"] < 0] = 0

        pricing["Call Price"] = list(call_option_ev(sim_, price, price/moneyness, r) 
                                 for sim_, price, r in zip(simulator, self.underlying_data["Adj Close"][self.split_date:], pricing["Annualized Rate"]))
        pricing["Put Price"] = list(put_option_ev(sim_, price, price/moneyness, r) 
                                 for sim_, price, r in zip(simulator, self.underlying_data["Adj Close"][self.split_date:], pricing["Annualized Rate"]))

        # Dropping dates outside forecast
        pricing = pricing.dropna(axis=0, subset="Call Price")

        # Calculating Implied Volatility
        pricing["Call IV"] = black_scholes_merton.implied_volatility.implied_volatility(
            pricing["Call Price"],
            pricing["Underlying Price"],
            pricing["Underlying Price"]/moneyness,
            simulator.period/252,
            pricing["Annualized Rate"]-1,
            ["c"],
            pricing["Dividend Yield"],
            return_as="numpy"
        )
        pricing["Put IV"] = black_scholes_merton.implied_volatility.implied_volatility(
            pricing["Put Price"],
            pricing["Underlying Price"],
            pricing["Underlying Price"]/moneyness,
            simulator.period/252,
            pricing["Annualized Rate"]-1,
            ["p"],
            pricing["Dividend Yield"],
            return_as="numpy"
        )

        #Shifting to expiry date
        if shift:
            pricing.set_index("Expiry Date").rename(columns={"index":"Date"})

        
        return pricing

    def plot_conditional_vol_rolling(self, period:int):
        """
        Plot rolling conditional volatility
        @param period: Period to average instantaneous conditional volatility over
        """
        plt.plot(self.res.conditional_volatility.rolling(window=period).mean()*np.sqrt(252)/100)

class ARIMAForecastModel(EconForecastModel):
    def __init__(self,ticker:str, start:datetime, end:datetime, split_date:datetime, volatility_period:int, order:tuple, seasonal_order:tuple=(0,0,0,0)):
        """
        Initializes an ARIMA Forecast Model Object
        @param ticker: The ticker of the underlying to be modelled
        @param start: The start date of the training data used in the model
        @param end: The end date of the data to be modelled
        @param split_date: The end date of the training data, and the start date of the data to be modelled
        @param order: The ARIMA order to be used
        @param seasonal_order: The ARIMA seasonal order to be used
        @param volatility_period: The time period used to calculate the volatility
        """
        # If dates provided are strings, convert them to datetimes
        if isinstance(start, str):
            start = datetime.datetime.strptime(start,"%Y-%m-%d")
        if isinstance(split_date, str):
            split_date = datetime.datetime.strptime(split_date,"%Y-%m-%d")
        if isinstance(end, str):
            end = datetime.datetime.strptime(end, "%Y-%m-%d")

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
        self.dividends = yf.Ticker(ticker).dividends/100
        self.dividends.index = self.dividends.index.tz_convert(None)
        self.rates = distributions.get_risk_free_rate(None, None)
        
        #Training model
        self.model = pm.arima.ARIMA(
            order=order,
            seasonal_order=seasonal_order
        )
        self.res = self.model.fit(self.underlying_returns[:split_date])
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
        forecasted_volatility = np.std(self.forecast, axis=1, ddof=0)*np.sqrt(252)
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
        sim = np.array([self.model.fit(self.underlying_data["Log Returns"][indices[0]]).arima_res_.simulate(
            nsimulations=period, repetitions=1000, anchor=len(self.underlying_data["Log Returns"][:self.split_date])+i).T 
            for i, indices in enumerate(data_generator)])
        return Simulation(sim, period, 1, self.split_date)
    
    def options_pricing(self, simulator, moneyness, shift=True):
        """
        Returns Dataframe with call, and put, forecasted pricing (present value), and realized value (present value)
        using monte carlo estimation with the passed simulator object
        @param simulator: Simulator object to be used for estimation
        @param moneyness: fraction into the money the option is
        @param shift: whether to shift the date to the expiry date (True) or leave it at date
        of forecast (optional)
        @return: Dataframe
        """
        #Creating dataframe with pricing
        pricing = pd.DataFrame({
            "Underlying Price": self.underlying_data["Adj Close"][self.split_date:]},
            index = self.underlying_data["Adj Close"][self.split_date:][:simulator.length].index
        )

        # Creating Expiry date column
        pricing["Expiry Date"] = pricing.index + datetime.timedelta(days=simulator.period)

        # Applying discount
        pricing = pricing.join(self.rates, how="left")

        # Joining Dividend Yields
        dividends = self.dividends.reindex(
            index = pd.date_range(self.dividends.index[0], self.dividends.index[-1]),
            method = "ffill"
        )
        dividends.index = dividends.index.date
        pricing = pricing.join(dividends, how="left").rename(columns={"Dividends":"Dividend Yield"})

        # Finding realized values and fixing realized values to not contain negative values
        # We fill because options expire across weekends, not only trading days
        data = yf.download(self.ticker, self.start, self.end + datetime.timedelta(days=simulator.period))
        filled_underlying_data = data["Adj Close"].reindex(
            index = pd.date_range(self.start, self.end + datetime.timedelta(days=simulator.period)),
            method = "ffill"
        )
        pricing = pricing.join(filled_underlying_data.shift(-simulator.period), how="left").rename(columns={"Adj Close":"Expired Underlying Price"}) 
        pricing["Strike Price"] = pricing["Underlying Price"]/moneyness
        pricing["Realized Call Value"] = (pricing["Expired Underlying Price"] - pricing["Strike Price"])/(pricing["Daily Rate"]**simulator.period)
        pricing["Realized Put Value"] = (-pricing["Expired Underlying Price"] + pricing["Strike Price"])/(pricing["Daily Rate"]**simulator.period)
        pricing["Realized Call Value"][pricing["Realized Call Value"] < 0] = 0
        pricing["Realized Put Value"][pricing["Realized Put Value"] < 0] = 0

        pricing["Call Price"] = list(call_option_ev(sim_, price, price/moneyness, r) 
                                 for sim_, price, r in zip(simulator, self.underlying_data["Adj Close"][self.split_date:], pricing["Annualized Rate"]))
        pricing["Put Price"] = list(put_option_ev(sim_, price, price/moneyness, r) 
                                 for sim_, price, r in zip(simulator, self.underlying_data["Adj Close"][self.split_date:], pricing["Annualized Rate"]))

        # Dropping dates outside forecast
        pricing = pricing.dropna(axis=0, subset="Call Price")

        # Calculating Implied Volatility
        pricing["Call IV"] = black_scholes_merton.implied_volatility.implied_volatility(
            pricing["Call Price"],
            pricing["Underlying Price"],
            pricing["Underlying Price"]/moneyness,
            simulator.period/252,
            pricing["Annualized Rate"]-1,
            ["c"],
            pricing["Dividend Yield"],
            return_as="numpy"
        )
        pricing["Put IV"] = black_scholes_merton.implied_volatility.implied_volatility(
            pricing["Put Price"],
            pricing["Underlying Price"],
            pricing["Underlying Price"]/moneyness,
            simulator.period/252,
            pricing["Annualized Rate"]-1,
            ["p"],
            pricing["Dividend Yield"],
            return_as="numpy"
        )

        #Shifting to expiry date
        if shift:
            pricing.set_index("Expiry Date").rename(columns={"index":"Date"})
        
        return pricing

class ARIMAARCHForecastModel(EconForecastModel):
    def __init__(self, ticker:str, start:datetime, end:datetime, split_date:datetime, volatility_period:int, p:int, q:int, model:str, order:tuple, seasonal_order:tuple=(0,0,0,0)):
        """
        Initializes an ARIMA+GARCH Forecast Model Object
        @param ticker: The ticker of the underlying to be modelled
        @param start: The start date of the training data used in the model
        @param end: The end date of the data to be modelled
        @param split_date: The end date of the training data, and the start date of the data to be modelled
        @param volatility_period: The time period used to calculate the volatility
        @param p: GARCH Model p
        @param q: GARCH Model q
        @param model: Model used for volatility process (GARCH, EGARCH)
        @param order: The ARIMA order to be used
        @param seasonal_order: The ARIMA seasonal order to be used
        """
        # If dates provided are strings, convert them to datetimes
        if isinstance(start, str):
            start = datetime.datetime.strptime(start,"%Y-%m-%d")
        if isinstance(split_date, str):
            split_date = datetime.datetime.strptime(split_date,"%Y-%m-%d")
        if isinstance(end, str):
            end = datetime.datetime.strptime(end, "%Y-%m-%d")

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
        self.arch_model = ARCHModel(pd.DataFrame({"Residuals":(self.arima_model.res.predict_in_sample()) - self.underlying_data["Log Returns"]}, index=self.underlying_data.index).replace(np.nan, 0), split_date, self.volatility_period, self.model, self.p, self.q)
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
        return Simulation(sim, period, 1, self.split_date)
    
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
            "Underlying Price": self.underlying_data["Adj Close"][self.split_date:]},
            index = self.underlying_data["Adj Close"][self.split_date:][:-simulator.period+1].index
        )

        # Creating Expiry date column
        pricing["Expiry Date"] = pricing.index + datetime.timedelta(days=simulator.period)

        # Applying discount
        rates = self.arima_model.rates
        pricing = pricing.join(rates, how="left")

        # Joining Dividend Yields
        dividends = self.arima_model.dividends.reindex(
            index = pd.date_range(self.arima_model.dividends.index[0], self.arima_model.dividends.index[-1]),
            method = "ffill"
        )
        dividends.index = dividends.index.date
        pricing = pricing.join(dividends, how="left").rename(columns={"Dividends":"Dividend Yield"})

        # Finding realized values and fixing realized values to not contain negative values
        # We fill because options expire across weekends, not only trading days
        data = yf.download(self.ticker, self.start, self.end + datetime.timedelta(days=simulator.period))
        filled_underlying_data = data["Adj Close"].reindex(
            index = pd.date_range(self.start, self.end + datetime.timedelta(days=simulator.period)),
            method = "ffill"
        )
        pricing = pricing.join(filled_underlying_data.shift(-simulator.period), how="left").rename(columns={"Adj Close":"Expired Underlying Price"}) 
        pricing["Strike Price"] = pricing["Underlying Price"]/moneyness
        pricing["Realized Call Value"] = (pricing["Expired Underlying Price"] - pricing["Strike Price"])/(pricing["Daily Rate"]**simulator.period)
        pricing["Realized Put Value"] = (-pricing["Expired Underlying Price"] + pricing["Strike Price"])/(pricing["Daily Rate"]**simulator.period)
        pricing["Realized Call Value"][pricing["Realized Call Value"] < 0] = 0
        pricing["Realized Put Value"][pricing["Realized Put Value"] < 0] = 0

        pricing["Call Price"] = list(call_option_ev(sim_, price, price/moneyness, r) 
                                 for sim_, price, r in zip(simulator, self.underlying_data["Adj Close"][self.split_date:], pricing["Annualized Rate"]))
        pricing["Put Price"] = list(put_option_ev(sim_, price, price/moneyness, r) 
                                 for sim_, price, r in zip(simulator, self.underlying_data["Adj Close"][self.split_date:], pricing["Annualized Rate"]))

        # Calculating Implied Volatility
        pricing["Call IV"] = black_scholes_merton.implied_volatility.implied_volatility(
            pricing["Call Price"],
            pricing["Underlying Price"],
            pricing["Underlying Price"]/moneyness,
            simulator.period/252,
            pricing["Annualized Rate"]-1,
            ["c"],
            pricing["Dividend Yield"],
            return_as="numpy"
        )
        pricing["Put IV"] = black_scholes_merton.implied_volatility.implied_volatility(
            pricing["Put Price"],
            pricing["Underlying Price"],
            pricing["Underlying Price"]/moneyness,
            simulator.period/252,
            pricing["Annualized Rate"]-1,
            ["p"],
            pricing["Dividend Yield"],
            return_as="numpy"
        )

        #Shifting to expiry date
        if shift:
            pricing.set_index("Expiry Date").rename(columns={"index":"Date"})

        return pricing

class HistoricVolModel(EconForecastModel):
    def __init__(self, ticker:str, start:datetime, end:datetime, split_date:datetime, volatility_period:int, historic_r_period:int, historic_vol_period:int):
        """
        Initializes an Historical Volatility based Model Object
        @param ticker: The ticker of the underlying to be modelled
        @param start: The start date of the training data used in the model
        @param end: The end date of the data to be modelled
        @param split_data: The end date of the training data, and the start date of the data to be modelled
        @param volatility_period: The time period used to calculate the volatility
        """
        # If dates provided are strings, convert them to datetimes
        if isinstance(start, str):
            start = datetime.datetime.strptime(start,"%Y-%m-%d")
        if isinstance(split_date, str):
            split_date = datetime.datetime.strptime(split_date,"%Y-%m-%d")
        if isinstance(end, str):
            end = datetime.datetime.strptime(end, "%Y-%m-%d")

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
        self.dividends = yf.Ticker(ticker).dividends/100
        self.dividends.index = self.dividends.index.tz_convert(None)
        self.rates = distributions.get_risk_free_rate(None, None)

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
        return Simulation(sim, period, 1, self.split_date)
    
    def options_pricing(self, simulator, moneyness, shift=True):
        """
        TODO: Implement black-scholes pricing if simulator is None
        Returns Dataframe with call, and put, forecasted pricing
        using monte carlo estimation with the passed simulator object
        @param simulator: (optional) simulator object to be used for estimation, if None, the model will use the black scholes formula to price the options
        @param moneyness: fraction into the money the option is
        @param shift: whether to shift the date to the expiry date (True) or leave it at date
        of forecast (optional)
        @return: Dataframe
        """
        #Creating dataframe with pricing
        pricing = pd.DataFrame({
            "Underlying Price": self.underlying_data["Adj Close"][self.split_date:]},
            index = self.underlying_data["Adj Close"][self.split_date:].index
        )

        # Creating Expiry date column
        pricing["Expiry Date"] = pricing.index + datetime.timedelta(days=simulator.period)

        # Applying discount
        pricing = pricing.join(self.rates, how="left")

        # Joining Dividend Yields
        dividends = self.dividends.reindex(
            index = pd.date_range(self.dividends.index[0], self.dividends.index[-1]),
            method = "ffill"
        )
        dividends.index = dividends.index.date
        pricing = pricing.join(dividends, how="left").rename(columns={"Dividends":"Dividend Yield"})

        # Finding realized values and fixing realized values to not contain negative values
        # We fill because options expire across weekends, not only trading days
        data = yf.download(self.ticker, self.start, self.end + datetime.timedelta(days=simulator.period))
        filled_underlying_data = data["Adj Close"].reindex(
            index = pd.date_range(self.start, self.end + datetime.timedelta(days=simulator.period)),
            method = "ffill"
        )
        pricing = pricing.join(filled_underlying_data.shift(-simulator.period), how="left").rename(columns={"Adj Close":"Expired Underlying Price"}) 
        pricing["Strike Price"] = pricing["Underlying Price"]/moneyness
        pricing["Realized Call Value"] = (pricing["Expired Underlying Price"] - pricing["Strike Price"])/(pricing["Daily Rate"]**simulator.period)
        pricing["Realized Put Value"] = (-pricing["Expired Underlying Price"] + pricing["Strike Price"])/(pricing["Daily Rate"]**simulator.period)
        pricing["Realized Call Value"][pricing["Realized Call Value"] < 0] = 0
        pricing["Realized Put Value"][pricing["Realized Put Value"] < 0] = 0

        pricing["Call Price"] = list(call_option_ev(sim_, price, price/moneyness, r) 
                                 for sim_, price, r in zip(simulator, self.underlying_data["Adj Close"][self.split_date:], pricing["Annualized Rate"]))
        pricing["Put Price"] = list(put_option_ev(sim_, price, price/moneyness, r) 
                                 for sim_, price, r in zip(simulator, self.underlying_data["Adj Close"][self.split_date:], pricing["Annualized Rate"]))

        # Dropping dates outside forecast
        pricing = pricing.dropna(axis=0, subset="Call Price")

        # Calculating Implied Volatility
        pricing["Call IV"] = black_scholes_merton.implied_volatility.implied_volatility(
            pricing["Call Price"],
            pricing["Underlying Price"],
            pricing["Underlying Price"]/moneyness,
            simulator.period/252,
            pricing["Annualized Rate"]-1,
            ["c"],
            pricing["Dividend Yield"],
            return_as="numpy"
        )
        pricing["Put IV"] = black_scholes_merton.implied_volatility.implied_volatility(
            pricing["Put Price"],
            pricing["Underlying Price"],
            pricing["Underlying Price"]/moneyness,
            simulator.period/252,
            pricing["Annualized Rate"]-1,
            ["p"],
            pricing["Dividend Yield"],
            return_as="numpy"
        )

        #Shifting to expiry date
        if shift:
            pricing.set_index("Expiry Date").rename(columns={"index":"Date"})
        
        return pricing