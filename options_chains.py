import pandas as pd
import numpy as np
from statistics import median

def read_options_csv(filename, symbol, start=None, end=None):
    """
    Reads csv and ouptuts dataframe with options chain
    @param filename: Filename containing options chain csv
    @param symbol: (optional) Symbol of underlying
    @param start: (optional) Start date of options contract data
    @param end: (optional) End date of options contract data
    @returns: Dataframe
    """
    # Read and format csv
    options_chain = pd.read_csv(filename)
    options_chain['expiration'] = pd.to_datetime(options_chain['expiration'])
    options_chain = options_chain.astype({
        'strike':'float',
        'bid':'float',
        'ask':'float',
        'vol':'float',
        'delta':'float',
        'gamma':'float',
        'theta':'float',
        'vega':'float',
        'rho':'float'
        })
    
    options_chain = options_chain[options_chain["act_symbol"] == symbol]
    if start is not None:
        options_chain = options_chain[options_chain["date"] >= start]
    if end is not None:
        options_chain = options_chain[options_chain["date"] <= end]
    
    return options_chain

class OptionsChain:
    def __init__(self, ticker: str, year: int):
        """
        Creates options chain object
        param:ticker: Ticker
        param:year: Year
        attr:data : Dataframe Object with EOD data of option chain for given ticker over given year
        """
        #Download option chain data to dataframe
        options = pd.DataFrame()
        for quarter in range(1, 5):
            for part in range(1, 4):
                df = pd.read_csv(f"options_chain_data/{ticker}_eod_{year}/spy_eod_{year}{'%02d' % (part + (3*(quarter-1)))}.txt")
                df = df.set_index(df.columns[1])
                df = df.rename(
                    columns = lambda x: x.replace(" ", "").replace("[", "").replace("]", "").replace("_"," ")
                    )
                df = df.drop(columns = df.columns[[0,5]])
                options = pd.concat((options,df))

        #Give better NA format
        options = options.replace(" ", np.nan)

        #dtype fix
        for col in ["C IV", "P IV", "C DELTA", "P DELTA"]:
            options[col] = pd.to_numeric(options[col])
        
        self.data = options

    def ask_dist(self, dte: int, min_moneyness: float, max_moneyness: float):
        """
        Returns dataframe with minimum, median, and maximum ask prices for puts and calls in
        the options chain given they are of a certain maturity period and moneyness
        :param dte: Days till expiry
        :param min_moneyness: Minimum moneyness of option
        :param max_moneyness: Maximum moneyness of option
        :return: Dataframe with index as expiry data
        """
        spy_ask = self.data[(self.data["DTE"] == dte) & (self.data["C DELTA"] >= min_moneyness) & (self.data["C DELTA"] <= max_moneyness)]\
        .groupby("EXPIRE DATE").agg(
        {"C ASK":[lambda x: max(x.dropna()), lambda x: min(x.dropna()), lambda x: median(x.dropna())], 
        "P ASK": [lambda x: max(x.dropna()), lambda x: min(x.dropna()), lambda x: median(x.dropna())]}
        )
        spy_ask.index = pd.to_datetime(spy_ask.index)
        spy_ask.columns = spy_ask.columns.set_levels(["min","max","median"], level=1)

        return spy_ask
    
    def bid_dist(self, dte: int, min_moneyness: float, max_moneyness: float):
        """
        Returns dataframe with minimum, median, and maximum bid prices for puts and calls in
        the options chain given they are of a certain maturity period and moneyness
        :param dte: Days till expiry
        :param min_moneyness: Minimum moneyness of option
        :param max_moneyness: Maximum moneyness of option
        :return: Dataframe with index as expiry date
        """
        spy_bid = self.data[(self.data["DTE"] == dte) & (self.data["C DELTA"] >= min_moneyness) & (self.data["C DELTA"] <= max_moneyness)]\
        .groupby("EXPIRE DATE").agg(
        {"C BID":[lambda x: max(x.dropna()), lambda x: min(x.dropna()), lambda x: median(x.dropna())], 
        "P BID": [lambda x: max(x.dropna()), lambda x: min(x.dropna()), lambda x: median(x.dropna())]}
        )
        spy_bid.index = pd.to_datetime(spy_bid.index)
        spy_bid.columns = spy_bid.columns.set_levels(["min","max","median"], level=1)

        return spy_bid

    def iv_dist(self, dte: int, min_moneyness: float, max_moneyness: float):
        """
        Returns dataframe with minimum, median, and maximum IV values for puts and calls in
        the options chain given they are of a certain maturity period and moneyness
        :param dte: Days till expiry
        :param min_moneyness: Minimum moneyness of option
        :param max_moneyness: Maximum moneyness of option
        :return: Dataframe with index as expiry date
        """
        spy_iv = self.data[(self.data["DTE"] == dte) & (self.data["C DELTA"] >= min_moneyness) & (self.data["C DELTA"] <= max_moneyness)]\
        .groupby("EXPIRE DATE").agg(
        {"C IV":[lambda x: max(x.dropna()), lambda x: min(x.dropna()), lambda x: median(x.dropna())], 
        "P IV": [lambda x: max(x.dropna()), lambda x: min(x.dropna()), lambda x: median(x.dropna())]}
        )
        spy_iv.index = pd.to_datetime(spy_iv.index)
        spy_iv.columns = spy_iv.columns.set_levels(["min","max","median"], level=1)

        return spy_iv