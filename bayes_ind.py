from datetime import datetime, timedelta
import numpy as np
import bayesloop as bl
import sympy.stats as stats
from tqdm import tqdm_notebook

import pandas as pd
from urllib.parse import urlencode

import aiohttp
import config


def get_points(index, n=8):
    points = []
    for i in range(n):
        points.append(int(len(index.data) * i / (n - 1)))
    return points


def log_std(returns):  # std dev of log returns
    return np.log(returns).std()


def log_returns(df):  # return log returns of prices
    logPrices = np.log(df)
    logReturns = np.diff(logPrices)
    return logReturns


def data_to_array(dataframe):  # return array of close prices
    df = dataframe.filter(['close'])
    df.close = df.close.astype(float)
    # df = df.close[1:].values
    prices = np.array(df)
    return prices


# function to calculate log standard deviation of returns (volatility)
# for prices using a rolling window x minutes wide, 1 day = 390 minutes
def rolling_log_std(prices, x=389):
    logReturns = log_returns(prices)
    logReturns = logReturns[-x:]
    return log_std(logReturns)


def create_bayesian_study(close_prices):
    len_prices = len(close_prices)
    lg_returns = rolling_log_std(close_prices, x=len_prices)
    try:
        sigma = log_std(close_prices)
    except:
        sigma = 0.006
    S = bl.OnlineStudy(storeHistory=True)
    L = bl.om.ScaledAR1('rho', bl.oint(-1, 1, 100),
                        'sigma', bl.oint(0, sigma, 800))
    S.set(L)

    T1 = bl.tm.CombinedTransitionModel(
        bl.tm.GaussianRandomWalk('s1', bl.cint(0, 1.5e-01, 15), target='rho',
                                 prior=stats.Exponential('expon', 1. / 3.0e-02)),
        bl.tm.GaussianRandomWalk('s2', bl.cint(0, 1.5e-04, 50), target='sigma')
    )
    T2 = bl.tm.Independent()
    S.add('normal', T1)
    S.add('chaotic', T2)
    #  1 news announcement per day
    S.setTransitionModelPrior([(len_prices - 1) / len_prices, 1 / len_prices])
    for r in tqdm_notebook(lg_returns):
        S.step(r)

    return r, sigma


def get_data(symbol):
    """
    Get data from Tiingo API
    Returns a numpy array of the 1min close prices for 1 day
    """
    API_KEY = config.TIINGO_API_KEY
    symbol = symbol.lower()
    TIINGO_BATCH_API = "https://api.tiingo.com/iex/{}/prices?".format(symbol)
    params = {
        'columns': 'date,open,high,low,close,volume',
        'startDate': (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d'),
        'endDate': (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d'),
        'resampleFreq': '1min',
        'token': API_KEY
    }
    url = TIINGO_BATCH_API + urlencode(params)
    with aiohttp.ClientSession() as session:
        with session.get(url) as response:
            json_response = response.json()
            try:
                df = pd.DataFrame(json_response)
                df.set_index('date', inplace=True)
                # df['date'] = pd.to_datetime(df['date'])
                df['symbol'] = symbol
                df = df[['symbol', 'open', 'high', 'low', 'close', 'volume']]
                return df
            except Exception as e:
                print(e)
                return pd.DataFrame()

    return df
    # df = df.filter(['close'])
    # df.close = df.close.astype(float)
    # df = df.close[1:].values


if __name__ == '__main__':
    data = get_data(symbol=config.STOCK_SYMBOL)
    prices = data_to_array(data)
    create_bayesian_study(prices)
    # get points
