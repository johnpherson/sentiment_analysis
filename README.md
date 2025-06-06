# WallStreetBets Sentiment Analysis

This web application analyzes sentiment from r/wallstreetbets subreddit posts and compares it with real-time stock prices to identify potential investment opportunities.

## Features
- Reddit sentiment analysis using Langchain with Claude
- Real-time stock price tracking
- Interactive visualization of sentiment vs price
- Identification of stocks with high sentiment but low price movement

## Setup
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your API credentials:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent
ANTHROPIC_API_KEY=your_anthropic_api_key
```
4. Run the application:
```bash
streamlit run app.py
```

## Usage
1. Enter a stock symbol in the search box
2. View the sentiment analysis results and price correlation
3. Analyze the trends and potential opportunities

## Technologies Used
- Langchain with Claude for NLP and sentiment analysis
- PRAW (Python Reddit API Wrapper)
- yfinance for stock data
- Streamlit for web interface
- Plotly for interactive visualizations 