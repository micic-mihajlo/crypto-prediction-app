# data.py
import yfinance as yf
import pandas as pd
import requests
import streamlit as st

@st.cache_data
def fetch_price_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

@st.cache_data
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