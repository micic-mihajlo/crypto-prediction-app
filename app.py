# app.py
import streamlit as st
from data import fetch_price_data, fetch_news_data
from sentiment import perform_sentiment_analysis
from model import train_model, create_dataset
from risk import calculate_volatility, calculate_sharpe_ratio
from backtesting import backtest_strategy
import numpy as np
from datetime import datetime, timedelta

def main():
    st.title("Cryptocurrency Price Prediction and Analysis")

    # Sidebar inputs
    ticker = st.sidebar.text_input("Enter cryptocurrency ticker (e.g., BTC-USD)", "BTC-USD")
    start_date = st.sidebar.date_input("Start date", datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input("End date", datetime.now())
    lookback = st.sidebar.slider("Lookback period (days)", 1, 60, 30)
    risk_free_rate = st.sidebar.slider("Risk-free rate (%)", 0.0, 10.0, 2.0) / 100

    if ticker:
        # Fetch and preprocess data
        with st.spinner("Fetching data..."):
            price_data = fetch_price_data(ticker, start_date, end_date)
            news_data = fetch_news_data(ticker.split("-")[0], start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

        # Display historical price chart
        st.subheader("Historical Price Chart")
        st.line_chart(price_data["Close"])

        # Train the model
        with st.spinner("Training model..."):
            model, scaler = train_model(price_data["Close"].values, lookback)

        # Predict the price for the next day
        last_price = price_data["Close"].iloc[-1]
        last_data = scaler.transform(price_data["Close"].values[-lookback:].reshape(-1, 1))
        predicted_price = model.predict(last_data.reshape(1, lookback, 1))
        predicted_price = scaler.inverse_transform(predicted_price)[0][0]

        st.subheader("Price Prediction")
        st.write(f"Last Price: ${last_price:.2f}")
        st.write(f"Predicted Price (Next Day): ${predicted_price:.2f}")

        # Perform sentiment analysis
        with st.spinner("Analyzing sentiment..."):
            news_data["sentiment"] = news_data["title"].apply(perform_sentiment_analysis)

        st.subheader("Sentiment Analysis")
        sentiment_score = news_data["sentiment"].mean()
        st.write(f"Sentiment Score: {sentiment_score:.2f}")

        # Calculate volatility and Sharpe ratio
        volatility = calculate_volatility(price_data["Close"], window=30)
        sharpe_ratio = calculate_sharpe_ratio(price_data["Close"], risk_free_rate, window=30)

        st.subheader("Risk Management")
        st.write(f"Volatility (30-day): {volatility.iloc[-1]:.4f}")
        st.write(f"Sharpe Ratio (30-day): {sharpe_ratio.iloc[-1]:.2f}")

        # Backtest a simple moving average strategy
        sma_strategy = lambda data: data.rolling(window=30).mean() > data.rolling(window=90).mean()
        portfolio_value = backtest_strategy(price_data["Close"], strategy_func=sma_strategy, initial_capital=10000)

        st.subheader("Backtesting")
        st.line_chart(portfolio_value)

    st.warning("Disclaimer: This application is for educational purposes only and should not be used for making actual financial decisions.")

if __name__ == "__main__":
    main()