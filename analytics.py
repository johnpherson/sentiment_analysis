import pandas as pd
import numpy as np
from collections import Counter
from textblob import TextBlob
import re
from datetime import datetime, timedelta

def calculate_sentiment_momentum(sentiment_df, window=3):
    """Calculate sentiment momentum using rolling averages"""
    daily_sentiment = sentiment_df.groupby(
        sentiment_df['date'].dt.date
    )['sentiment'].mean()
    
    momentum = daily_sentiment.rolling(window=window).mean().diff()
    return momentum

def extract_mentioned_stocks(text):
    """Extract stock symbols from text using regex"""
    # Match $TICKER or common stock mention patterns
    pattern = r'\$([A-Z]{1,5})|(?<![\w$])([A-Z]{1,5})(?=\s|$)'
    matches = re.findall(pattern, text.upper())
    # Flatten matches and remove empty strings
    tickers = [tick for match in matches for tick in match if tick]
    return list(set(tickers))

def analyze_correlated_stocks(posts):
    """Find stocks frequently mentioned together"""
    stock_mentions = []
    for post in posts:
        text = f"{post['title']} {post['body']}"
        mentioned = extract_mentioned_stocks(text)
        if len(mentioned) > 1:
            stock_mentions.extend(list(zip(mentioned[:-1], mentioned[1:])))
    
    return Counter(stock_mentions)

def calculate_sentiment_price_correlation(sentiment_df, price_df):
    """Calculate correlation between sentiment and price movements"""
    # Resample both dataframes to daily frequency
    daily_sentiment = sentiment_df.groupby(
        sentiment_df['date'].dt.date
    )['sentiment'].mean()
    
    daily_returns = price_df['Close'].pct_change()
    
    # Align the dates and convert to sorted list
    common_dates = sorted(list(set(daily_sentiment.index) & set(daily_returns.index)))
    sentiment = daily_sentiment[common_dates]
    returns = daily_returns[common_dates]
    
    return sentiment.corr(returns)

def identify_sentiment_divergence(sentiment_df, price_df, threshold=0.5):
    """Identify periods where sentiment and price diverge significantly"""
    daily_sentiment = sentiment_df.groupby(
        sentiment_df['date'].dt.date
    )['sentiment'].mean()
    
    daily_returns = price_df['Close'].pct_change()
    
    # Normalize both series
    norm_sentiment = (daily_sentiment - daily_sentiment.mean()) / daily_sentiment.std()
    norm_returns = (daily_returns - daily_returns.mean()) / daily_returns.std()
    
    # Calculate divergence
    divergence = norm_sentiment - norm_returns
    
    # Find significant divergences
    significant = divergence[abs(divergence) > threshold]
    
    return significant

def analyze_post_impact(posts, price_df):
    """Analyze how high-impact posts correlate with price movements"""
    high_impact_posts = []
    
    for post in posts:
        # Consider posts with high engagement
        if post['score'] > 100 or post['num_comments'] > 50:
            post_date = post['created_utc'].date()
            
            # Get next day's return if available
            try:
                next_day = post_date + timedelta(days=1)
                if next_day in price_df.index:
                    next_day_return = (
                        price_df.loc[next_day, 'Close'] - 
                        price_df.loc[post_date, 'Close']
                    ) / price_df.loc[post_date, 'Close']
                    
                    high_impact_posts.append({
                        'date': post_date,
                        'title': post['title'],
                        'score': post['score'],
                        'comments': post['num_comments'],
                        'sentiment': post['sentiment'],
                        'next_day_return': next_day_return
                    })
            except:
                continue
    
    return pd.DataFrame(high_impact_posts)

def get_trading_signals(sentiment_df, price_df, sentiment_threshold=0.6):
    """Generate trading signals based on sentiment and price action"""
    daily_sentiment = sentiment_df.groupby(
        sentiment_df['date'].dt.date
    )['sentiment'].mean()
    
    signals = []
    for date in daily_sentiment.index:
        if date in price_df.index:
            sentiment = daily_sentiment[date]
            price = price_df.loc[date, 'Close']
            volume = price_df.loc[date, 'Volume']
            
            signal = {
                'date': date,
                'sentiment': sentiment,
                'price': price,
                'volume': volume,
                'signal': 'NEUTRAL'
            }
            
            # Generate signals based on sentiment and volume
            if sentiment > sentiment_threshold and volume > price_df['Volume'].mean():
                signal['signal'] = 'STRONG_BUY'
            elif sentiment > sentiment_threshold:
                signal['signal'] = 'BUY'
            elif sentiment < -sentiment_threshold and volume > price_df['Volume'].mean():
                signal['signal'] = 'STRONG_SELL'
            elif sentiment < -sentiment_threshold:
                signal['signal'] = 'SELL'
                
            signals.append(signal)
    
    return pd.DataFrame(signals) 