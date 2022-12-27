"""https://www.pythonforfinance.net/2019/04/19/multi-threading-trading-strategy-back-tests-and-monte-carlo
-simulations-in-python/ """
import datetime as datetime
import itertools
import os
import time
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool as Pool
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import pystore
import requests
from matplotlib import pyplot as plt

import config

# import random as rd

# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split

PATH = 'data'
if not os.path.exists(PATH):
    os.makedirs(PATH)
pystore.set_path(PATH)
store = pystore.store('datastore')

# this list is purely designed to generate gradient colors for the plot
colorlist = config.COLORLIST


def get_tiingo_data(sym, freq):
    # load data, save to parquet
    """
    Get data from Tiingo API
    Returns a numpy array of the 1min close prices for 1 day
    """
    API_KEY = config.TIINGO_API_KEY

    sym = sym.lower()
    TIINGO_BATCH_API = "https://api.tiingo.com/iex/{}/prices?".format(sym)
    params = {
        'columns': 'date,open,high,low,close,volume',
        'startDate': (datetime.today() - timedelta(days=4)).strftime('%Y-%m-%d'),
        'endDate': (datetime.today() - timedelta(days=4)).strftime('%Y-%m-%d'),
        'resampleFreq': freq,
        'token': API_KEY
    }
    url = TIINGO_BATCH_API + urlencode(params)
    # print(url)
    response = requests.get(url).json()
    df = pd.DataFrame(response, columns='date open close high low'.split())  # volume'.split())
    df = df.astype(
        {'date': 'datetime64[ms]', 'open': float, 'high': float, 'low': float, 'close': float})  # , 'volume': float})
    # df = df.filter(['close'])
    # df = df.close[1:].values
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # save to parquet
    # ticker = symbol.upper()
    # collection = store.collection('NASDAQ')
    # if collection.item(ticker).get().exists:
    #     collection.append(f'{str(ticker)}', df, metadata={'source': 'Tiingo Iex'})
    # else:
    #     collection.write(f'{str(ticker)}', df, metadata={'source': 'Tiingo Iex'})

    # item = collection.item(f'{str(ticker)}')
    # d_data = item.data  # <-- Dask dataframe (see dask.pydata.org)
    # data['symbol'] = str(symbol)
    # print last 4 rows of df
    # print(df.tail(4))
    return df


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(itertools.islice(it, size)), ())


# function to calculate Sharpe Ratio - Risk free rate element excluded for simplicity
def sharpe_ratio(returns, n=390):
    try:
        return np.sqrt(n) * (returns.mean() / returns.std())
    except ZeroDivisionError:
        return 0


def ma_strat(df, short_ma, long_ma):
    # create columns with MA values

    df['short_ma'] = np.round(df['close'].rolling(window=short_ma).mean(), 2)
    df['long_ma'] = np.round(df['close'].rolling(window=long_ma).mean(), 2)
    # create column with moving average spread differential
    df['short_ma-long_ma'] = df['short_ma'] - df['long_ma']
    # set desired number of points as threshold for spread difference and create column containing strategy 'Stance'
    X = 5
    df['Stance'] = np.where(df['short_ma-long_ma'] > X, 1, 0)
    df['Stance'] = np.where(df['short_ma-long_ma'] < -X, -1, df['Stance'])
    df['Stance'].value_counts()
    # create columns containing daily market log returns and strategy daily log returns
    df['Market Returns'] = np.log(df['close'] / df['close'].shift(1))
    df['Strategy'] = df['Market Returns'] * df['Stance'].shift(1)
    # set strategy starting equity to 1 (i.e. 100%) and generate equity curve
    df['Strategy Equity'] = df['Strategy'].cumsum()
    # calculate Sharpe Ratio
    # try/except to escape case of division by zero
    try:
        sharpe = sharpe_ratio(df['Strategy'])
    except ZeroDivisionError:
        sharpe = 0
    return df['Strategy'].cumsum(), sharpe, df['Strategy'].mean(), df['Strategy'].std()


def monte_carlo_strat(df, inputs, iters):
    # set number of days for each Monte Carlo simulation
    timesteps = len(df)

    # use the current inputs to backtest the strategy and record
    # various results metrics
    perf, sharpe, mu, sigma = ma_strat(df, inputs[0], inputs[1])
    # print(inputs[0])
    # create two empty lists to store results of MC simulation
    mc_results = []
    mc_results_final_val = []

    # run the specified number of MC simulations and store relevant results
    for j in range(iters):
        daily_returns = np.random.normal(mu, sigma, timesteps) + 1
        price_list = [1]
        for x in daily_returns:
            price_list.append(price_list[-1] * x)

        # store the individual price path for each simulation
        mc_results.append(price_list)
        # store only the ending value of each individual price path
        mc_results_final_val.append(price_list[-1])

    return (inputs, perf, sharpe, mu, sigma, mc_results, mc_results_final_val)


def filter_best_fit(df, mc_results, iters):
    # the one with the least standard deviation wins
    # find the index of the MC simulation with the smallest standard deviation from df['close']
    # and use this index to select the corresponding price path from mc_results
    # select pick as mc_results where min(mc_results - df['close'])
    # make data numpy array for easier manipulation
    df_filtered = df.loc[:, ['Close']]
    # print(df_filtered, 'df_filtered')
    pick = []
    for j in range(iters):
        pick = np.where(np.subtract(mc_results, df_filtered) == np.min(np.subtract(mc_results, df_filtered)))

    # std = float('inf')
    # pick = 0
    # print(len(mc_results), len(df))
    # for counter in range(iters):
    #
    #     temp = np.std(np.subtract(mc_results[counter][:len(df)], df['close']))
    #     if temp < std:
    #         std = temp,
    #         pick = counter
    return pick


def parallel_monte_carlo(df, inputs, iterations):
    pool = Pool(5)
    future_res = [pool.apply_async(monte_carlo_strat, args=(df, inputs[i], iterations)) for i in range(len(inputs))]
    samples = [f.get() for f in future_res]
    print(samples, 'samples')

    return samples


# result plotting
def plot(df, mc_res, pick, ticker):
    # plot every simulation, mc_res (a list of lists)
    # and highlight the best fit with the actual dataset, data['close']
    ax = plt.figure(figsize=(10, 5)).add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i in range(int(len(mc_res))):
        if i != pick:
            ax.plot(df.index[:len(df)], mc_res[i][:len(df)], alpha=0.05)
        ax.plot(df.index[:len(df)], mc_res[pick][:len(df)], c='#5398d9', linewidth=5, label='Best Fitted')
    df['close'].iloc[:len(df)].plot(c='#d75b66', linewidth=5, label='Actual')
    plt.title(f'Monte Carlo Simulation\nTicker: {ticker}')
    plt.legend(loc=0)
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.show()


if __name__ == '__main__':
    # read in price data
    # data = pd.read_csv('F.csv', index_col='Date', parse_dates=True)
    symbol = config.STOCK_SYMBOL  # str(input('Symbol to get historical data, e.g "tsla"') or 'tsla')
    Freq = config.FREQUENCY
    # start_Date = str(input('Limits metrics to on or after the startDate (>=). '
    #                        'Parameter MUST be in YYYY-MM-DD format. Default is one year from now.'))
    data = get_tiingo_data(symbol, Freq)

    # generate our list of possible short window length inputs
    short_mas = np.linspace(20, 50, 30, dtype=int)

    # generate our list of possible long window length inputs
    long_mas = np.linspace(100, 200, 30, dtype=int)

    # generate a list of tuples containing all combinations of
    # long and short window length possibilities
    mas_combined = list(itertools.product(short_mas, long_mas))

    # use our helper function to split the moving average tuples list
    # into slices of length 180
    mas_combined_split = list(chunk(mas_combined, 180))

    # set required number of MC simulations per backtest optimisation
    iters = 40  # 2000

    # start timer
    start_time = time.time()

    # call our multithreading function
    # results, mc_results, pick = parallel_monte_carlo(data, mas_combined_split, iters)
    results = parallel_monte_carlo(data, mas_combined_split, iters)

    # pick = filter_best_fit(data, results[mc_results], iters)

    # plot
    # plot(data, mc_results, pick, symbol)

    # print number of seconds the process took
    print("MP--- %s seconds for para---" % (time.time() - start_time))
