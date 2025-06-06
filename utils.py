import os
import praw
import yfinance as yf
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

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

def get_stock_data(symbol, period='1mo'):
    stock = yf.Ticker(symbol)
    history = stock.history(period=period)
    return history

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
    except:
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