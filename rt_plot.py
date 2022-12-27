#!/usr/bin/env python3
import math
import time
from datetime import datetime
from urllib.parse import urlencode

import finplot as fplt
import pandas as pd
import requests
import exchange_calendars as xcals
import config
from tqdm import tqdm, trange

nys = xcals.get_calendar("XNYS")  # New York Stock Exchange


def cumcnt_indices(v):
    v[~v] = math.nan
    cumsum = v.cumsum().fillna(method='pad')
    reset = -cumsum[v.isnull()].diff().fillna(cumsum)
    r = v.where(v.notnull(), reset).cumsum().fillna(0.0)
    return r.astype(int)


def td_sequential(close):
    close4 = close.shift(4)
    td = cumcnt_indices(close > close4)
    ts = cumcnt_indices(close < close4)
    return td, ts


def update():
    # load data
    # limit = 500
    start_time = int(time.time() * 1000) - (500 - 2) * 60 * 1000
    start = datetime.fromtimestamp(start_time / 1000).strftime('%Y-%m-%d')
    print(start)
    # check if market is open now and if not wait until it opens
    # seconds to open

    while not nys.is_session(start):
        seconds_to_open = nys.next_open(datetime.now()).timestamp() - datetime.now().timestamp()
        minutes_to_open = seconds_to_open / 60
        start_time = int(time.time() * 1000) - (500 - 2) * 60 * 1000
        start = datetime.fromtimestamp(start_time / 1000).strftime('%Y-%m-%d')
        if seconds_to_open > 0:
            print('Market is closed. Waiting until it opens.', f'{minutes_to_open:.0f} minutes to go.')
            with tqdm(total=100) as pbar:  # progress bar
                for i in range(100):  # 100 steps
                    time.sleep(seconds_to_open / 100)
                    pbar.update(1)
    else:
        start = datetime.fromtimestamp(start_time / 1000).strftime('%Y-%m-%d')
        print(start, 'market is open, loading data')
    """
    Get data from Tiingo API
    Returns a numpy array of the 1min close prices for 1 day
    """
    API_KEY = config.TIINGO_API_KEY
    symbol = config.STOCK_SYMBOL
    symbol = symbol.lower()
    TIINGO_BATCH_API = "https://api.tiingo.com/iex/{}/prices?".format(symbol)
    params = {
        'columns': 'date,open,high,low,close,volume',
        'startDate': start,  # (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d'),
        # 'endDate': '2022-12-23',  # (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d'),
        'resampleFreq': '1min',
        # 'format': 'pandas',
        'token': API_KEY
    }
    url = TIINGO_BATCH_API + urlencode(params)
    # print(url)
    response = requests.get(url).json()
    df = pd.DataFrame(response, columns='date open close high low volume'.split())
    df = df.astype(
        {'date': 'datetime64[ms]', 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})

    # calculate indicator
    tdup, tddn = td_sequential(df['close'])
    df['tdup'] = [('%i' % i if 0 < i < 10 else '') for i in tdup]
    df['tddn'] = [('%i' % i if 0 < i < 10 else '') for i in tddn]

    # pick columns for our three data sources: candlesticks and TD sequencial labels for up/down
    candlesticks = df['date open close high low'.split()]
    volumes = df['date open close volume'.split()]
    td_up_labels = df['date high tdup'.split()]
    td_dn_labels = df['date low tddn'.split()]
    if not plots:
        # first time we create the plots
        try:
            global ax
            plots.append(fplt.candlestick_ochl(candlesticks))
            plots.append(fplt.volume_ocv(volumes, ax=ax.overlay()))
            plots.append(fplt.labels(td_up_labels, color='#009900'))
            plots.append(fplt.labels(td_dn_labels, color='#990000', anchor=(0.5, 0)))
        except Exception as e:
            print(e)
    else:
        # every time after we just update the data sources on each plot
        try:
            plots[0].update_data(candlesticks)
            plots[1].update_data(volumes)
            plots[2].update_data(td_up_labels)
            plots[3].update_data(td_dn_labels)
        except Exception as e:
            print(e)


plots = []
ax = fplt.create_plot(f'Realtime {config.STOCK_SYMBOL} 1m Sequential (Tiingo REST)', init_zoom_periods=100,
                      maximize=False)
update()
fplt.timer_callback(update, 5.0)  # update (using synchronous rest call) every N seconds

fplt.show()
