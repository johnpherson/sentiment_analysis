import os
import praw
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time
import pandas as pd
import json
from pathlib import Path
import requests
import numpy as np
from requests.exceptions import RequestException
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import yfinance as yf

load_dotenv()

# Create cache directory if it doesn't exist
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_stock_data(symbol, period='1mo'):
    """Get stock data from cache if available and not expired"""
    cache_file = CACHE_DIR / f"{symbol}_{period}.json"
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                cache_time = datetime.fromtimestamp(cached_data['timestamp'])
                # Cache expires after 1 hour
                if datetime.now() - cache_time < timedelta(hours=1):
                    df = pd.DataFrame(cached_data['data'])
                    df.index = pd.to_datetime(df['Date'])
                    df = df.drop('Date', axis=1)
                    return df
        except Exception as e:
            print(f"Cache read error: {e}")
            # Delete invalid cache file
            cache_file.unlink(missing_ok=True)
    return None

def save_stock_data_to_cache(symbol, period, data):
    """Save stock data to cache"""
    cache_file = CACHE_DIR / f"{symbol}_{period}.json"
    try:
        # Reset index to include Date as a column and convert to records
        df_to_save = data.copy()
        df_to_save.index.name = 'Date'
        df_to_save = df_to_save.reset_index()
        
        # Convert datetime objects to strings for JSON serialization
        df_to_save['Date'] = df_to_save['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        cache_data = {
            'timestamp': datetime.now().timestamp(),
            'data': df_to_save.to_dict('records')
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    except Exception as e:
        print(f"Cache write error: {e}")

def get_reddit_client():
    return praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )

def get_wsb_posts(symbol, limit=100):
    reddit = get_reddit_client()
    subreddit = reddit.subreddit('wallstreetbets')
    
    # Search for posts containing the stock symbol
    query = f'({symbol} OR ${symbol})'
    posts = subreddit.search(query, limit=limit, time_filter='month')
    
    return [{
        'title': post.title,
        'body': post.selftext,
        'created_utc': datetime.fromtimestamp(post.created_utc),
        'score': post.score,
        'num_comments': post.num_comments
    } for post in posts]

def robust_json_parse(response_text: str) -> Dict[str, Any]:
    """
    Robust JSON parser that handles common API response issues
    """
    try:
        # First try normal JSON parsing
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Initial JSON decode failed: {e}")
        
        # Try to clean and fix common issues
        cleaned_text = response_text.strip()
        
        # Remove any non-JSON content before the actual JSON
        json_start = cleaned_text.find('{')
        if json_start != -1:
            cleaned_text = cleaned_text[json_start:]
        
        # Try parsing again
        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            print("JSON parsing failed completely")
            raise ValueError(f"Unable to parse JSON response: {response_text[:200]}...")

class StockResponse(BaseModel):
    """Pydantic model for validating stock API response structure"""
    meta_data: Optional[Dict[str, str]] = Field(None, alias="Meta Data")
    time_series_daily: Optional[Dict[str, Dict[str, str]]] = Field(None, alias="Time Series (Daily)")
    error_message: Optional[str] = Field(None, alias="Error Message")
    note: Optional[str] = Field(None, alias="Note")

def validate_stock_response(raw_data: Dict[str, Any]) -> StockResponse:
    """Validate and parse stock API response using Pydantic"""
    try:
        parser = PydanticOutputParser(pydantic_object=StockResponse)
        # Convert dict to JSON string for the parser
        json_string = json.dumps(raw_data)
        return parser.parse(json_string)
    except Exception as e:
        print(f"Pydantic validation failed: {e}")
        # Fallback: create StockResponse manually
        return StockResponse(
            meta_data=raw_data.get("Meta Data"),
            time_series_daily=raw_data.get("Time Series (Daily)"),
            error_message=raw_data.get("Error Message"),
            note=raw_data.get("Note")
        )

def get_stock_data(symbol, period='1mo'):
    """Get stock data using yfinance with caching"""
    # Try to get from cache first
    cached_data = get_cached_stock_data(symbol, period)
    if cached_data is not None:
        print(f"Using cached data for {symbol}")
        return cached_data

    try:
        # Calculate date range based on period
        end_date = datetime.now()
        if period.endswith('d'):
            days = int(period[:-1])
            start_date = end_date - timedelta(days=days)
        else:  # Default to 30 days if period format is not recognized
            start_date = end_date - timedelta(days=30)
            
        # Download data using yfinance
        print(f"\nFetching data from Yahoo Finance...")
        
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Standardize column names and format
        data.columns = data.columns.get_level_values(0)  # Remove multi-level columns if present
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = data[expected_columns]  # Select only the columns we need
            
        # Cache successful result
        save_stock_data_to_cache(symbol, period, data)
        print(f"Successfully fetched data for {symbol}")
        return data
            
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        print("Falling back to dummy data...")
        
        # Generate dummy data with past dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate realistic dummy data
        base_price = 100
        dummy_data = []
        for i, date in enumerate(dates):
            # Add some randomness to make it look realistic
            variation = np.random.normal(0, 2)
            open_price = base_price + variation
            high_price = open_price + abs(np.random.normal(0, 1))
            low_price = open_price - abs(np.random.normal(0, 1))
            close_price = open_price 
            volume = int(np.random.normal(1000000, 200000))
            
            dummy_data.append({
                'Open': round(open_price, 2),
                'High': round(high_price, 2),
                'Low': round(low_price, 2),
                'Close': round(close_price, 2),
                'Volume': max(volume, 100000)  # Ensure positive volume
            })
            base_price = close_price  # Carry forward for next day
        
        dummy_df = pd.DataFrame(dummy_data, index=dates)
        return dummy_df

def analyze_sentiment(text):
    template = """
    You are analyzing stock market sentiment. Rate the following text on a scale from -1 to 1.
    -1 means very negative/bearish
    0 means neutral
    1 means very positive/bullish
    Only respond with a single number between -1 and 1.

    Text: {text}
    Score:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["text"])
    # Using tinyllama, which is much more lightweight than mistral
    llm = Ollama(
        model="tinyllama",
        temperature=0,
        num_ctx=512  # Smaller context window to reduce memory usage
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    
    try:
        result = chain.run(text=text)
        # Clean up the result to get just the number
        result = result.strip().split('\n')[0]
        return float(result)
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        return 0

def calculate_average_sentiment(posts):
    sentiments = []
    for post in posts:
        # Combine title and body for sentiment analysis
        text = f"{post['title']} {post['body']}"
        sentiment = analyze_sentiment(text)
        sentiments.append({
            'date': post['created_utc'],
            'sentiment': sentiment,
            'score': post['score']
        })
    
    return sentiments