# sentiment.py
from textblob import TextBlob

def perform_sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity