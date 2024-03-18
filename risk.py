# risk.py
import numpy as np

def calculate_volatility(data, window):
    returns = data.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(window)
    return volatility

def calculate_sharpe_ratio(data, risk_free_rate, window):
    returns = data.pct_change()
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.rolling(window=window).mean() / calculate_volatility(data, window)
    return sharpe_ratio