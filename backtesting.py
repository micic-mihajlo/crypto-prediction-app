# backtesting.py
def backtest_strategy(data, strategy_func, initial_capital):
    positions = strategy_func(data)
    portfolio_value = initial_capital * (1 + positions.shift(1) * data.pct_change()).cumprod()
    return portfolio_value