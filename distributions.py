import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import acf, pacf
import datetime

def deannualize(annual_rate: float, periods=365):
    """
    Returns the deannualized rate for an annual rate
    @param annual_rate: the annual rate
    @param periods: (optional) amount of periods to segment annualized rate to
    @return: deannualized rate
    """
    return (annual_rate) ** (1/periods)

def get_risk_free_rate(start: str, end:str):
    """
    Returns dataframe with risk free rates daily and annualized
    Rates based on 3-Month US Treasury Bills
    @param start: start date of rates to be downloaded non-inclusive
    @param end: end date of rates to be downloaded non-inclusive
    @return: datafram with risk free rates
    """
    # Download 3-Month Treasury Bills
    annualized_rates = 1.0 + (yf.download("^IRX", start=start, end=end)["Adj Close"]/100)

    # De-annualize
    daily_rates = annualized_rates.apply(deannualize)

    # Return dataframe
    return pd.DataFrame({"Annualized Rate": annualized_rates, "Daily Rate": daily_rates}) 

def discount(value, start: str, end: str):
    """
    Returns the discounted value of the input value at start time given the input value 
    is the value at end time
    Discount is based on 3-Month US Treasury Bills
    @param value: undiscounted value
    @param start: time for the discounted value
    @param end: time at the undiscounted value non-inclusive
    @return: the discounted value
    """
    rates = get_risk_free_rate(start, end)
    return value / np.prod(np.array(rates["Daily Rate"]))

def get_returns(tickers: list[str], start: str, end: str, min=None):
    """
    Returns dataframe with the returns data for securities in the list of tickers
    :param tickers: list of tickers
    :param start: start date for data to be downloaded
    :param end: end date for data to be downloaded
    :param min: the minimum number of entries for a security, otherwise it is dropped from the dataframe
        (optional)
    :return: dataframe with the returns data
    """
    # Downloading data
    data = yf.download(tickers, start=start, end=end)

    # Calculating log-returns
    returns = np.log1p(data["Adj Close"].pct_change())
    if min is not None:
        returns = returns.dropna(axis=1, thresh=len(returns)-min)
    return returns

def get_realized_option_value(ticker, start: str, end: str, moneyness: float, dte: int):
    """
    Returns a dataframe with present realized option value (call and put) given
    a ticker and the start and end date of the data to be collected
    (NB Value is discounted)
    @param prices: Dataframe with adjusted close of underlying
    @param moneyness: moneyness of the option
    @param dte: Days till expiry of the option
    @return: Dataframe
    """
    # Creating filled adjusted close
    # Because options mature across non-trading days
    data = yf.download(ticker, start, end)
    fill_data = yf.download(ticker, start, datetime.datetime.strptime(end,"%Y-%m-%d") + datetime.timedelta(days=dte+1))
    filled_underlying_data = fill_data["Adj Close"].reindex(
            index = pd.date_range(fill_data.index[0], fill_data.index[-1]),
            method = "ffill"
        ).rename("Underlying")

    # Adding Expiry Dates
    data["Expiry Date"] = data.index + datetime.timedelta(days=dte)
    
    # Adding moneyness and days till expiry to dataframe
    data["Moneyness"] = moneyness
    data["DTE"] = dte

    # Adding tau
    data["Tau"] = dte/252

    # Adding continuous annual dividend yield
    dividends = yf.Ticker(ticker).dividends/100
    dividends.index = dividends.index.tz_convert(None)
    dividends = dividends.reindex(
        index = pd.date_range(dividends.index[0], dividends.index[-1]),
        method = "ffill"
    )
    dividends.index = dividends.index.date
    data = data.join(dividends, how="left").rename(columns={"Dividends":"Dividend Yield"})

    # Adding risk free rate
    rates = get_risk_free_rate(None, None)

    # Joining risk free rate
    data = data.join(rates, how="left")

    # Joining expired underlying price and strike price
    data = data.join(filled_underlying_data.shift(-dte), how="left").rename(columns={"Underlying":"Expired Underlying Price"})
    data["Strike Price"] = data["Adj Close"]/moneyness
    data["Realized Call Value"] = (data["Expired Underlying Price"] - data["Strike Price"])/(data["Daily Rate"]**dte)
    data["Realized Put Value"] = (-data["Expired Underlying Price"] + data["Strike Price"])/(data["Daily Rate"]**dte)

    # Fixing to remove negative values
    data["Realized Call Value"][data["Realized Call Value"] < 0] = 0
    data["Realized Put Value"][data["Realized Put Value"] < 0] = 0

    return data


def get_realized_volatility(returns, period=1):
    """
    Returns the volatility of the given series within the given instantaneous time periods
    :param returns: Series of the returns of a security
    :param period: Time period to calculate volatility over
    """
    # Return rolling window standard deviation
    return returns.rolling(window=period).std(ddof=0)*np.sqrt(252)

def returns_distribution(tickers: list[str], start: str, end:str, filename=None, header=None):
    """
    Finds returns distribution for stock data, and tests hypothesis of normally distributed returns
    with constant variance
    :param tickers: The ticker symbol for data to be downloaded
    :param start: The start date for downloaded stock data
    :param end: The end date for downloaded stock data
    :param filename: The name of the file for the data to be uploaded to (optional)
    :param header: The header configuration for the file (optional)
    :return: Dataframe with the number of observations, mean return, standard deviation of returns,
    skewness of returns, kurtosis of returns, and k-squared test p-value of the returns
    """
    # Download data
    data = yf.download(tickers, start=start, end=end)
    returns = np.log1p(data["Close"].pct_change())
    returns = returns.dropna(axis=1, thresh=len(returns)-200)
    
    # Construct distribution dataframe
    symbols = returns.columns
    sample_stats = pd.DataFrame({"Observations": (len(returns[symbol]) for symbol in symbols),
                      "Mean Return": list(np.mean(returns, axis=0)),
                      "Standard Deviation": list(np.std(returns, axis=0)),
                      "Skewness": list(stats.skew(returns, axis=0, nan_policy='omit')),
                      "Kurtosis": list(stats.kurtosis(returns, axis=0, nan_policy='omit')),
                      "K-Squared P-Value": list(stats.normaltest(returns, axis=0, nan_policy='omit').pvalue)},
                      index = symbols)
    
    # Save to dataframe if filename is provided
    if filename is not None:
        sample_stats.to_csv(filename, header=header)
    
    return sample_stats

def returns_autocorrelation(tickers: list[str], start:str, end:str, filename=None, header=None):
    """
    Returns three dataframes of autocorrelation, ljung-box q statistic, and p-value for returns
    of stocks in the list of tickers
    :param tickers: list of tickers
    :param start: start date to download data from
    :param end: end date to download data from
    :param filename: name of file to upload data to (optional)
    :param header: header configuration of the file (optional)
    :return: Three dataframes, autocorrelation, ljung-box q statistic, and p-value
    """
    # Download data
    data = yf.download(tickers, start=start, end=end)
    returns = np.log1p(data["Adj Close"].pct_change())
    returns = returns.dropna(axis=1, thresh=len(returns)-200)

    # Formatting function
    def ac_format(ac):
        lags = ac[0][1:6]
        rest = ac[1:3]
        return [lags] + list(rest)

    # Collecting autocorrelation statistics
    full_r_ac = np.array(list(ac_format(acf(returns[symbol].dropna(), nlags=5, qstat=True)) for symbol in returns.columns))
    full_sqr_r_ac = np.array(list(ac_format(acf(returns[symbol].dropna()**2, nlags=5, qstat=True)) for symbol in returns.columns))
    full_abs_r_ac = np.array(list(ac_format(acf(abs(returns[symbol].dropna()), nlags=5, qstat=True)) for symbol in returns.columns))
    r_pac = np.array(list(pacf(returns[symbol].dropna(), nlags=5)[1:6] for symbol in returns.columns)).T
    sqr_r_pac = np.array(list(pacf(returns[symbol].dropna()**2, nlags=5)[1:6] for symbol in returns.columns)).T
    abs_r_pac = np.array(list(pacf(abs(returns[symbol].dropna()), nlags=5)[1:6] for symbol in returns.columns)).T
    
    r_ac = full_r_ac[:,0].T
    r_ldq = full_r_ac[:,1].T
    r_pval = full_r_ac[:,2].T
    sqr_r_ac = full_sqr_r_ac[:,0].T
    sqr_r_ldq = full_sqr_r_ac[:,1].T
    sqr_r_pval = full_sqr_r_ac[:,2].T
    abs_r_ac = full_abs_r_ac[:,0].T
    abs_r_ldq = full_abs_r_ac[:,1].T
    abs_r_pval = full_abs_r_ac[:,2].T

    corr = pd.DataFrame(
    {
    **{("Returns Autocorrelation", f"Lag {l+1}"): r_ac[l] for l in range(5)},
    **{("Square Returns Autocorrelation", f"Lag {l+1}"): sqr_r_ac[l] for l in range(5)},
    **{("Absolute Returns Autocorrelation", f"Lag {l+1}"): abs_r_ac[l] for l in range(5)},
    **{("Returns Partial Autocorrelation", f"Lag {l+1}"): r_pac[l] for l in range(5)},
    **{("Square Returns Partial Autocorrelation", f"Lag {l+1}"): sqr_r_pac[l] for l in range(5)},
    **{("Absolute Returns Partial Autocorrelation", f"Lag {l+1}"): abs_r_pac[l] for l in range(5)}
    }, index = returns.columns)

    ldq = pd.DataFrame(
    {
    **{("Returns Ljung-Box Q Statistic", f"Lag {l+1}"): r_ldq[l] for l in range(5)},
    **{("Squared Returns Ljung-Box Q Statistic", f"Lag {l+1}"): sqr_r_ldq[l] for l in range(5)},
    **{("Absolute Returns Ljung-Box Q Statistic", f"Lag {l+1}"): abs_r_ldq[l] for l in range(5)}
    }, index = returns.columns)

    p = pd.DataFrame(
    {
    **{("Returns Autocorrelation P Value", f"Lag {l+1}"): r_pval[l] for l in range(5)},
    **{("Squared Returns Autocorrelation P Value", f"Lag {l+1}"): sqr_r_pval[l] for l in range(5)},
    **{("Absolute Returns Autocorrelation P Value", f"Lag {l+1}"): abs_r_pval[l] for l in range(5)}
    }, index = returns.columns)

    if filename is not None:
        corr.to_csv("corr-"+filename, header=header)
        p.to_csv("p-"+filename, header=header)

    return corr, ldq, p