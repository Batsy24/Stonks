import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import datetime

ticker_input = 'gs'
raw = yf.download(ticker_input, period='max', progress=False)
scaler = MinMaxScaler()


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

    # log dates for plotting later
    date = df.index.to_list()

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
    cols = df.columns.to_list()
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=cols)

    return df, date


input_dat, dates = transform()  # use this dates variable in the Predictor script and pray really hard that it works.
time = len(input_dat) - 1


# weekly 64, daily 32
class WindowGenerator:
    def __init__(self, batch_size, input_width: int, label_width: int, offset: int,
                 df=input_dat, label_columns=None):

        # assertions
        assert input_width > 0, "Input width must be a positive number!"
        assert label_width > 0, "Label width must be a positive number!"
        assert offset > 0, "Offset value must be a positive number!"

        # DATAFRAMES
        self.df = df

        # giving indices to every column in label_columns (in this model there'll only be one label_column)
        # and to every column in the dataset
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_column_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(df.columns)}

        # WINDOW PARAMETERS.
        self.batch_size = batch_size

        # input parameters
        self.input_width = input_width
        self.label_width = label_width
        self.offset = offset
        self.total_window_width = input_width + offset

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_width)[self.input_slice]

        # label parameters
        self.label_start = self.total_window_width - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_width)[self.label_slice]

    def split_window(self, features):
        # (3, 7, 23)
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in
                               self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_width,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def input_df(self):
        return self.make_dataset(self.df)


def gen_data(batch_size, data=input_dat):
    n = 0

    wide_window = WindowGenerator(
        input_width=time, label_width=time, offset=1,
        label_columns=['Close'], batch_size=batch_size)

    dat = next(iter(wide_window.input_df))
    model_input = dat[0]
    index = wide_window.column_indices['Close']

    input_cols = model_input[n, :, index]

    return model_input, input_cols


model_inputs, input_cols = gen_data(32)
# 32 is just for test cuz I- tested this on daily a.i., 64 for weekly predictions


def load_models(path):
    model = tf.keras.models.load_model(path)
    return model


def get_predictions(model, data):
    predictions = model.predict(data)
    predictions = predictions[0, :, 0]

    return predictions


def inverse_transform(values, dates, future=None, index=3, cols=24):
    df_shaper = pd.DataFrame(values)

    for i in range(cols):
        df_shaper[i + 1] = df_shaper[0]

    df_shaper_columns = df_shaper.columns
    unscaled_val_arr = scaler.inverse_transform(df_shaper)
    df_unscaled = pd.DataFrame(unscaled_val_arr, columns=df_shaper_columns)
    inverse_values_df = df_unscaled[index]

    if future:
        for date in dates:
            date += datetime.timedelta(days=future)

        dates = dates[:-1]
        inverse_values_df.index = dates
        inverse_values = inverse_values_df.to_numpy()
    else:
        dates = dates[:-1]
        inverse_values_df.index = dates
        inverse_values = inverse_values_df.to_numpy()

    return inverse_values, inverse_values_df


# noinspection PyBroadException
def get_daily_forecast(prediction_df=None, input_df=None):
    forecast = prediction_df.iloc[-1:]
    past = input_df.iloc[:-1]

    return forecast, past


# noinspection PyBroadException
def get_2week_forecast(prediction_df=None, input_df=None):
    remove = -1 * 14

    forecast = prediction_df.iloc[remove:]
    past = input_df.iloc[:remove]

    return forecast, past


# make sure user can't input any of the input stock indices / have replacements

lstm = load_models('StonkNet-Daily copy')
predictions = get_predictions(lstm, model_inputs)

inputs, idf = inverse_transform(input_cols, dates)
predicted, pdf = inverse_transform(predictions, dates, future=14)
# future = 1, daily
# future = 14, weekly

# forecast, past = get_2week_forecast(prediction_df=pdf, input_df=idf)
forecast, past = get_daily_forecast(prediction_df=pdf, input_df=idf)

print(past.tail())
print(forecast.head())





