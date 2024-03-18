import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Function to fetch historical price data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_price_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

# Function to fetch news data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_news_data(query, start_date, end_date):
    url = f"https://newsapi.org/v2/everything?q={query}&from={start_date}&to={end_date}&sortBy=publishedAt&apiKey=1fae5448214f47b7a22e6f09117226dd"
    response = requests.get(url)
    data = response.json()
    if 'articles' in data:
        articles = data['articles']
        df = pd.DataFrame(articles, columns=["publishedAt", "title", "description"])
        df["publishedAt"] = pd.to_datetime(df["publishedAt"])
        return df
    else:
        return pd.DataFrame(columns=["publishedAt", "title", "description"])

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to create dataset for model training
def create_dataset(data, lookback):
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        Y.append(data[i + lookback])
    return np.array(X), np.array(Y)

# Function to train the LSTM model
@st.cache_resource  # Cache the model object
def train_model(data, lookback):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    X, Y = create_dataset(scaled_data, lookback)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, Y, epochs=10, batch_size=32, verbose=0)  # Set verbose to 0 to suppress training output

    return model, scaler

# Function to calculate volatility
@st.cache_data  # Cache the volatility calculation
def calculate_volatility(data, window):
    returns = data.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(window)
    return volatility

# Function to calculate Sharpe ratio
@st.cache_data  # Cache the Sharpe ratio calculation
def calculate_sharpe_ratio(data, risk_free_rate, window):
    returns = data.pct_change()
    excess_returns = returns - risk_free_rate / 252
    sharpe_ratio = excess_returns.rolling(window=window).mean() / calculate_volatility(data, window)
    return sharpe_ratio

# Function to backtest a strategy
@st.cache_data  # Cache the backtesting results
def backtest_strategy(data, strategy_func, initial_capital):
    positions = strategy_func(data)
    portfolio_value = initial_capital * (1 + positions.shift(1) * data.pct_change()).cumprod()
    return portfolio_value

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