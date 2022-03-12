# this is the most important script, it transforms raw data into data that the ai can
# process.you don't have any use for it YET


import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

ticker_input = 'gs'
raw = yf.download(ticker_input, period='max', progress=False)


def transform(df=raw):
    # fetching other indices
    jpm = yf.download('JPM', period='max', progress=False)
    ms = yf.download('MS', period='max', progress=False)
    ndaq = yf.download('^IXIC', period='max', progress=False)
    nkk = yf.download('^N225', period='max', progress=False)
    bse = yf.download('^BSESN', period='max', progress=False)
    nya = yf.download('^NYA', period='max', progress=False)

    # fetching vix data and cleaning
    url = 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv'
    vix = pd.read_csv(url, names=['Date', 'Open', 'High', 'Low', 'Close'])
    vix = vix.iloc[1:, :]

    # appending all data
    df['close ms'] = ms['Close']
    df['close jpm'] = jpm['Close']
    df['close ndaq'] = ndaq['Close']
    df['close nikkei'] = nkk['Close']
    df['close bse'] = bse['Close']
    df['close nya'] = nya['Close']

    vix = vix.set_index(vix['Date'])
    vix = vix.drop('Date', axis=1)
    vix.index = pd.to_datetime(vix.index)
    df.index = pd.to_datetime(df.index)

    df['close vix'] = vix['Close']
    df[['close vix']] = df[['close vix']].apply(pd.to_numeric)

    # calculate and append technical indicators
    df['ma20'] = df['Close'].rolling(window=20).mean()
    df['ma200'] = df['Close'].rolling(window=200).mean()

    df['26ema'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['12ema'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['MACD'] = (df['12ema'] - df['26ema'])  # MACD
    df['signal_line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['ema'] = df['Close'].ewm(com=0.5).mean()  # EMA....?

    df['momentum'] = df['Close'] - 1  # Momentum

    df['TP'] = (df['Close'] + df['Low'] + df['High']) / 3
    df['std'] = df['TP'].rolling(20).std(ddof=0)
    df['MA-TP'] = df['TP'].rolling(20).mean()
    df['BOLU'] = df['MA-TP'] + 2 * df['std']
    df['BOLD'] = df['MA-TP'] - 2 * df['std']

    df = df.drop('TP', axis=1)
    df = df.drop('MA-TP', axis=1)
    df = df.drop('std', axis=1)

    # clean up all data, replace NaNs etc
    mean_nkk = df['close nikkei'].mean()
    mean_bse = df['close bse'].mean()
    mean_vix = df['close vix'].mean()
    mean_bolu = df['BOLU'].mean()
    mean_bold = df['BOLD'].mean()

    mean_list = [mean_nkk, mean_bse, mean_vix, mean_bolu, mean_bold]  # , mean_ma7, mean_ma21]
    incomplete_cols = ['close nikkei', 'close bse', 'close vix', 'BOLU', 'BOLD']  # , 'ma7', 'ma21']
    n = 0

    for mean in mean_list:
        df[incomplete_cols[n]].fillna(value=mean, inplace=True)
        n += 1

    df['ma200'] = df['ma200'].fillna(0)
    df['ma20'] = df['ma20'].fillna(0)

    # adding fourier transform values to represent trend

    df['Date'] = df.index
    data = {'Date': df['Date'].tolist(),
            'Prices': df['Close'].tolist()}
    df_ = pd.DataFrame(data)
    df_.Date = pd.to_datetime(df_.Date)
    fft = np.fft.fft(np.asarray(df_['Prices'].tolist()))
    fft_df = pd.DataFrame({'fft': fft})
    fft_df['abs'] = np.abs(fft_df['fft'])
    df = df.drop('Date', axis=1)
    fft_list = np.asarray(fft_df['fft'].tolist())
    for num_ in [3, 6, 9, 50, 100]:
        fft_list_ = np.copy(fft_list)
        fft_list_[num_:-num_] = 0
        inverse_fft = np.abs(np.fft.ifft(fft_list_))

        if num_ == 3:
            df['long term'] = inverse_fft.tolist()
        elif num_ == 9:
            df['medium term'] = inverse_fft.tolist()

    # normalization
    index = df.index
    columns = df.columns.tolist()
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    return df


input_dat = transform()

# finalise model + normalise inverse

# search
# menu
