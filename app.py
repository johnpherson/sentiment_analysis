import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from utils import get_wsb_posts, get_stock_data, calculate_average_sentiment
from analytics import (
    calculate_sentiment_momentum,
    analyze_correlated_stocks,
    calculate_sentiment_price_correlation,
    identify_sentiment_divergence,
    analyze_post_impact,
    get_trading_signals
)
from wordcloud import WordCloud
import plotly.express as px

st.set_page_config(page_title="WSB Sentiment Analysis", layout="wide")

st.title("WallStreetBets Sentiment Analysis")

# Sidebar for user input
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., AAPL)", value="AAPL")
days = st.sidebar.slider("Days of History", min_value=7, max_value=30, value=30)
sentiment_threshold = st.sidebar.slider("Sentiment Threshold", min_value=0.0, max_value=1.0, value=0.6)
post_score_threshold = st.sidebar.slider("Minimum Post Score", min_value=1, max_value=1000, value=100)

# Compare with other stocks
compare_stocks = st.sidebar.text_input("Compare with other stocks (comma-separated)", value="")
comparison_symbols = [s.strip() for s in compare_stocks.split(",") if s.strip()] if compare_stocks else []

if symbol:
    with st.spinner(f"Analyzing {symbol}..."):
        # Get stock data
        stock_data = get_stock_data(symbol, period=f"{days}d")
        
        # Get Reddit posts and sentiment
        posts = get_wsb_posts(symbol, limit=100)
        sentiments = calculate_average_sentiment(posts)
        sentiment_df = pd.DataFrame(sentiments)
        
        # Calculate daily average sentiment
        daily_sentiment = sentiment_df.groupby(
            sentiment_df['date'].dt.date
        )['sentiment'].mean().reset_index()
        
        # Create main chart
        tab1, tab2, tab3 = st.tabs(["Price & Sentiment", "Advanced Analytics", "Trading Signals"])
        
        with tab1:
            # Create subplot with stock price and sentiment
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add stock price
            fig.add_trace(
                go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name=symbol
                ),
                secondary_y=False
            )
            
            # Add sentiment line
            fig.add_trace(
                go.Scatter(
                    x=daily_sentiment['date'],
                    y=daily_sentiment['sentiment'],
                    name="Sentiment",
                    line=dict(color='purple')
                ),
                secondary_y=True
            )
            
            # Add comparison stocks if specified
            for comp_symbol in comparison_symbols:
                comp_data = get_stock_data(comp_symbol, period=f"{days}d")
                fig.add_trace(
                    go.Scatter(
                        x=comp_data.index,
                        y=comp_data['Close'],
                        name=comp_symbol,
                        line=dict(dash='dot')
                    ),
                    secondary_y=False
                )
            
            fig.update_layout(
                title=f"{symbol} Stock Price vs WSB Sentiment",
                xaxis_title="Date",
                yaxis_title="Stock Price ($)",
                yaxis2_title="Sentiment Score",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Average Sentiment",
                    f"{sentiment_df['sentiment'].mean():.2f}",
                    delta=f"{sentiment_df['sentiment'].std():.2f} std"
                )
            
            with col2:
                price_change = ((stock_data['Close'][-1] - stock_data['Open'][0]) / 
                              stock_data['Open'][0] * 100)
                st.metric(
                    "Price Change",
                    f"{price_change:.2f}%",
                    delta=f"{stock_data['Close'][-1]:.2f} USD"
                )
            
            with col3:
                sentiment_momentum = calculate_sentiment_momentum(sentiment_df).iloc[-1]
                st.metric(
                    "Sentiment Momentum",
                    f"{sentiment_momentum:.2f}",
                    delta="Increasing" if sentiment_momentum > 0 else "Decreasing"
                )
            
            with col4:
                correlation = calculate_sentiment_price_correlation(sentiment_df, stock_data)
                st.metric(
                    "Sentiment-Price Correlation",
                    f"{correlation:.2f}",
                    delta="Strong" if abs(correlation) > 0.5 else "Weak"
                )
        
        with tab2:
            # Sentiment Momentum Chart
            momentum = calculate_sentiment_momentum(sentiment_df)
            fig_momentum = px.line(
                momentum,
                title="Sentiment Momentum",
                labels={"value": "Momentum", "index": "Date"}
            )
            st.plotly_chart(fig_momentum, use_container_width=True)
            
            # High Impact Posts Analysis
            st.subheader("High Impact Posts Analysis")
            impact_analysis = analyze_post_impact(posts, stock_data)
            if not impact_analysis.empty:
                impact_fig = px.scatter(
                    impact_analysis,
                    x="sentiment",
                    y="next_day_return",
                    size="score",
                    hover_data=["title", "comments"],
                    title="Post Impact Analysis"
                )
                st.plotly_chart(impact_fig, use_container_width=True)
            
            # Sentiment Divergence
            st.subheader("Sentiment-Price Divergence")
            divergences = identify_sentiment_divergence(sentiment_df, stock_data)
            if not divergences.empty:
                st.write("Periods of significant sentiment-price divergence:")
                st.dataframe(divergences)
            
            # Correlated Stocks
            st.subheader("Frequently Co-mentioned Stocks")
            correlations = analyze_correlated_stocks(posts)
            if correlations:
                st.write("Top stock pairs mentioned together:")
                for pair, count in correlations.most_common(5):
                    st.write(f"{pair[0]}-{pair[1]}: {count} times")
        
        with tab3:
            # Trading Signals
            signals_df = get_trading_signals(sentiment_df, stock_data, sentiment_threshold)
            
            # Create signals chart
            fig_signals = go.Figure()
            
            # Add price line
            fig_signals.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name="Price",
                    line=dict(color='gray')
                )
            )
            
            # Add buy and sell signals
            buy_signals = signals_df[signals_df['signal'].isin(['BUY', 'STRONG_BUY'])]
            sell_signals = signals_df[signals_df['signal'].isin(['SELL', 'STRONG_SELL'])]
            
            fig_signals.add_trace(
                go.Scatter(
                    x=buy_signals['date'],
                    y=buy_signals['price'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        color='green',
                        size=10,
                        symbol='triangle-up'
                    )
                )
            )
            
            fig_signals.add_trace(
                go.Scatter(
                    x=sell_signals['date'],
                    y=sell_signals['price'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        color='red',
                        size=10,
                        symbol='triangle-down'
                    )
                )
            )
            
            fig_signals.update_layout(
                title="Trading Signals Based on Sentiment and Volume",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400
            )
            
            st.plotly_chart(fig_signals, use_container_width=True)
            
            # Signal Details
            st.subheader("Signal Details")
            st.dataframe(signals_df)
        
        # Display recent posts
        st.subheader("Recent WSB Posts")
        filtered_posts = [p for p in posts if p['score'] >= post_score_threshold]
        for post in filtered_posts[:5]:
            with st.expander(f"{post['title']} (Score: {post['score']})"):
                st.write(post['body'])
                st.caption(f"Posted on: {post['created_utc']}")
                
        # Opportunity Analysis
        st.subheader("Investment Opportunity Analysis")
        avg_sentiment = sentiment_df['sentiment'].mean()
        price_trend = price_change
        
        if avg_sentiment > sentiment_threshold and price_trend < 5:
            st.success("🚀 High sentiment with relatively low price increase - Potential opportunity!")
        elif avg_sentiment < -sentiment_threshold and price_trend > 5:
            st.warning("⚠️ Low sentiment with price increase - Potential caution zone!")
        else:
            st.info("📊 Sentiment and price movement appear to be aligned") 