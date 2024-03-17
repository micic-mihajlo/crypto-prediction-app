import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

def create_dataset(data, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        Y.append(data[i+lookback])
    return np.array(X), np.array(Y)

def train_model(data, lookback):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, Y = create_dataset(scaled_data, lookback)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X, Y, epochs=100, batch_size=32, verbose=1)

    return model, scaler

def calculate_volatility(data, window):
    returns = data.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(window)
    return volatility

def calculate_sharpe_ratio(data, risk_free_rate, window):
    returns = data.pct_change()
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.rolling(window=window).mean() / calculate_volatility(data, window)
    return sharpe_ratio

def backtest_strategy(data, strategy_func, initial_capital):
    positions = strategy_func(data)
    portfolio_value = initial_capital * (1 + positions.shift(1) * data.pct_change()).cumprod()
    return portfolio_value