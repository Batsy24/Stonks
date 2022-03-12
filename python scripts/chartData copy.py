# another important script, gives you all kinds of graph data

import yfinance as yf
import pandas as pd
import numpy as np

ticker_input = 'gs'

# possible time periods: 1d, 7d, 1mo, 6mo, 1y, ytd, max
time_period_input = '6mo'

if time_period_input is not '1d':
    chart_data = yf.download(ticker_input, period=time_period_input, progress=False)
else:
    chart_data = yf.download(ticker_input, period=time_period_input, interval='1h', progress=False)


# window vals(long term) = 50, 200
# window vals(short term) = 7, 10, 20
def MA(window, dat=chart_data, plot_col='Close'):
    col_name = 'MA' + str(window)
    dat[col_name] = dat[plot_col].rolling(window=window).mean()

    return dat


# window vals(long term) = 50, 200
# window vals(short term) = 12, 26
def EMA(window, dat=chart_data, plot_col='Close'):
    col_name = 'EMA' + str(window)
    dat[col_name] = dat[plot_col].ewm(span=window, adjust=False).mean()

    return dat


def MACD(dat=chart_data, plot_col='Close'):
    dat['12ema'] = dat[plot_col].ewm(span=12, adjust=False).mean()
    dat['26ema'] = dat[plot_col].ewm(span=26, adjust=False).mean()

    dat['MACD'] = (dat['12ema'] - dat['26ema'])
    dat['signal_line'] = dat['MACD'].ewm(span=9, adjust=False).mean()

    dat = dat.drop('12ema')
    dat = dat.drop('26ema')

    return dat


# note: 1) look at end of this code for plotting code, use it as it is, text for help.
def Bollinger(dat=chart_data):
    dat['TP'] = (dat['Close'] + dat['Low'] + dat['High']) / 3
    dat['std'] = dat['TP'].rolling(20).std(ddof=0)
    dat['MA-TP'] = dat['TP'].rolling(20).mean()

    dat['BOLU'] = dat['MA-TP'] + 2 * dat['std']
    dat['BOLD'] = dat['MA-TP'] - 2 * dat['std']

    return dat

# =======================================================
# {{{NOTE FOR NISARG}}}
#
# i haven't like tested this code but im 99% sure the entire thing works but just in case do tell me if u face literally any issues
#
# ok so the thing is the first 2 functions are easy to plot u just gotta plt.plot(dat[col_name]) that's it,
# they handle x-axis too however, the 3rd and 4th code you need to know what it is in order to know how to plot it,
# I realise you probably don't have the time and so ive written some code for u for the last(4th) one and as for the 3rd one
# just know that it's the small graph u see below the big graph on the home page ui I sent u. The following two pieces of
# code are for plotting the 3rd and 4th func.
#


# this code is for plotting the values for the 4th function

# def plot_technical_indicators(dataset=df, last_days=5700): you can pass dataset=dat and last_days = len(dat)
#     plt.figure(figsize=(16, 10), dpi=100)
#     shape_0 = dataset.shape[0]
#     xmacd_ = shape_0 - last_days
#
#     dataset = dataset.iloc[-last_days:, :]
#     x_ = range(3, dataset.shape[0])
#     x_ = list(dataset.index)
#
#     plt.plot(dataset['BOLU'], label='Upper Band', color='c')
#     plt.plot(dataset['BOLD'], label='Lower Band', color='c')
#     plt.fill_between(x_, dataset['BOLD'], dataset['BOLU'], alpha=0.35)
#
#     plt.plot(dataset['Close'], label='Closing Price', color='b', alpha=0.25)
#     plt.title(f'Bollinger Bands')
#     plt.ylabel('USD')
#     plt.legend()
#     plt.show()

# -----------------------------------------------------------------------------------------------

# this function is for plotting the values for the 3rd function

# def plot_MACD_signal_line(dataset, last_days): # same as before, dataset=dat , last_day = len(dat)
#     shape_0 = dataset.shape[0]
#     xmacd_ = shape_0 - last_days
#
#     dataset = dataset.iloc[-last_days:, :]
#     x_ = range(3, dataset.shape[0])
#     x_ = list(dataset.index)
#     dataset[['MACD', 'signal_line']].plot(figsize=(16, 8))
#     dataset['Close'].plot(label='Closing Price', alpha=0.25, secondary_y=True)
#     plt.title('MACD and Signal')
