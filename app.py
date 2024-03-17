import streamlit as st
from data import fetch_price_data, preprocess_data, fetch_news_data, preprocess_news_data, perform_sentiment_analysis
from model import train_model, calculate_volatility, calculate_sharpe_ratio, backtest_strategy
import pandas as pd
import numpy as np

# Apply Streamlit caching to expensive operations like data fetching, preprocessing, and model training.
@st.cache(show_spinner=False, suppress_st_warning=True)
def fetch_and_preprocess_price_data(coin_id, days):
    btc_data = fetch_price_data(coin_id, days)
    btc_df = preprocess_data(btc_data)
    return btc_df

@st.cache(show_spinner=False, suppress_st_warning=True, allow_output_mutation=True)
def train_price_prediction_model(data, lookback):
    model, scaler = train_model(data, lookback)
    return model, scaler

@st.cache(show_spinner=False, suppress_st_warning=True)
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

    # Input for cryptocurrency selection
    coin_id = st.selectbox("Select Cryptocurrency", options=["bitcoin", "ethereum", "litecoin"], index=0)
    days_price_data = 365
    days_news_data = 30

    # Fetch and preprocess data
    btc_df = fetch_and_preprocess_price_data(coin_id, days_price_data)
    news_df = fetch_and_preprocess_news_data(coin_id, days_news_data)

    # Display historical price chart
    st.subheader("Historical Price Chart")
    st.line_chart(btc_df["Price"])

    # Train the model
    lookback = 30
    model, scaler = train_price_prediction_model(btc_df[["Price"]], lookback)

    # Predict the price for the next day
    last_data = scaler.transform(btc_df["Price"].values.reshape(-1, 1))[-lookback:]
    predicted_price = model.predict(last_data.reshape(1, lookback, 1))
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]

    st.subheader("Price Prediction")
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
