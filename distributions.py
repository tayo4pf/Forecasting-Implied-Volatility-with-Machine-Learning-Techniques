import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats

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
    data = yf.download(tickers, start=start, end=end)
    returns = np.log1p(data["Adj Close"].pct_change())
    if min is not None:
        returns = returns.dropna(axis=1, thresh=len(returns)-min)
    return returns

def get_realized_volatility(returns, period=1):
    """
    Returns the volatility of the given series within the given instantaneous time periods
    :param returns: Series of the returns of a security
    :param period: Time period to calculate volatility over
    """
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
    data = yf.download(tickers, start=start, end=end)
    returns = np.log1p(data["Close"].pct_change())
    returns = returns.dropna(axis=1, thresh=len(returns)-200)
    
    symbols = returns.columns
    sample_stats = pd.DataFrame({"Observations": (len(returns[symbol]) for symbol in symbols),
                      "Mean Return": list(np.mean(returns, axis=0)),
                      "Standard Deviation": list(np.std(returns, axis=0)),
                      "Skewness": list(stats.skew(returns, axis=0, nan_policy='omit')),
                      "Kurtosis": list(stats.kurtosis(returns, axis=0, nan_policy='omit')),
                      "K-Squared P-Value": list(stats.normaltest(returns, axis=0, nan_policy='omit').pvalue)},
                      index = symbols)
    
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
    data = yf.download(tickers, start=start, end=end)
    returns = np.log1p(data["Adj Close"].pct_change())
    returns = returns.dropna(axis=1, thresh=len(returns)-200)

    def ac_format(ac):
        lags = ac[0][1:6]
        rest = ac[1:3]
        return [lags] + list(rest)

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

    symbols = returns.columns
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