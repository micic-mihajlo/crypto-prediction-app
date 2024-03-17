import requests
import pandas as pd
from textblob import TextBlob

def fetch_price_data(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()
    return data

def preprocess_data(data):
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["Timestamp", "Price"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df.set_index("Timestamp", inplace=True)
    return df

def fetch_news_data(coin_name, days):
    url = f"https://newsapi.org/v2/everything?q={coin_name}&from={pd.Timestamp.now() - pd.Timedelta(days=days):%Y-%m-%d}&sortBy=publishedAt&apiKey=YOUR_NEWS_API_KEY"
    response = requests.get(url)
    data = response.json()
    return data

def preprocess_news_data(data):
    articles = data["articles"]
    df = pd.DataFrame(articles, columns=["publishedAt", "title", "description"])
    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    df.set_index("publishedAt", inplace=True)
    return df

def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity