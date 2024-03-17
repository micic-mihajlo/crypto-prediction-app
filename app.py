import streamlit as st
import pandas as pd
import numpy as np
from streamlit import cache
from data import fetch_price_data, preprocess_data, fetch_news_data, preprocess_news_data, perform_sentiment_analysis
from model import train_model, calculate_volatility, calculate_sharpe_ratio, backtest_strategy
from concurrent.futures import ThreadPoolExecutor

@cache
def fetch_and_preprocess_price_data(coin_id, days):
    btc_data = fetch_price_data(coin_id, days)
    btc_df = preprocess_data(btc_data)
    return btc_df

@cache
def train_price_prediction_model(data, lookback):
    model, scaler = train_model(data, lookback)
    return model, scaler

@cache
def fetch_and_preprocess_news_data(coin_name, days):
    news_data = fetch_news_data(coin_name, days)
    news_df = preprocess_news_data(news_data)
    return news_df

def sma_strategy(data, short_window=10, long_window=30):
    short_sma = data.rolling(window=short_window).mean()
    long_sma = data.rolling(window=long_window).mean()
    positions = np.where(short_sma > long_sma, 1, -1)
    return pd.Series(positions, index=data.index)

def main():
    st.title("Cryptocurrency Price Prediction and Analysis")
    st.write("This application predicts the price of Bitcoin (BTC) for the next day and provides sentiment analysis, risk management, and backtesting.")

    # Fetch and preprocess data in parallel
    with ThreadPoolExecutor() as executor:
        price_data_future = executor.submit(fetch_and_preprocess_price_data, "bitcoin", 365)
        news_data_future = executor.submit(fetch_and_preprocess_news_data, "bitcoin", 7)

    # Get the results from the futures
    btc_df = price_data_future.result()
    news_df = news_data_future.result()

    # Display historical price chart
    st.subheader("Historical Price Chart")
    st.line_chart(btc_df["Price"])

    # Train the model
    lookback = 30
    model, scaler = train_price_prediction_model(btc_df["Price"], lookback)

    # Predict the price for the next day
    last_price = btc_df["Price"].iloc[-1]
    last_data = scaler.transform(btc_df["Price"].values.reshape(-1, 1))[-lookback:]
    predicted_price = model.predict(last_data.reshape(1, lookback, 1))
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]

    st.subheader("Price Prediction")
    st.write(f"Last Price: ${last_price:.2f}")
    st.write(f"Predicted Price (Next Day): ${predicted_price:.2f}")

    # Perform sentiment analysis
    news_df["sentiment"] = news_df["title"].apply(perform_sentiment_analysis)

    st.subheader("Sentiment Analysis")
    sentiment_score = news_df["sentiment"].mean()
    st.write(f"Sentiment Score: {sentiment_score:.2f}")

    # Calculate volatility and Sharpe ratio
    volatility = calculate_volatility(btc_df["Price"], window=30)
    sharpe_ratio = calculate_sharpe_ratio(btc_df["Price"], risk_free_rate=0.02, window=30)

    st.subheader("Risk Management")
    st.write(f"Volatility (30-day): {volatility.iloc[-1]:.4f}")
    st.write(f"Sharpe Ratio (30-day): {sharpe_ratio.iloc[-1]:.2f}")

    # Backtest a simple moving average strategy
    portfolio_value = backtest_strategy(btc_df["Price"], strategy_func=sma_strategy, initial_capital=10000)

    st.subheader("Backtesting")
    st.line_chart(portfolio_value)

    st.warning("Disclaimer: This application is for educational purposes only and should not be used for making actual financial decisions.")

if __name__ == "__main__":
    main()