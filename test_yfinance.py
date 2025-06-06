import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

def test_stock_fetch(symbol="AAPL"):
    print(f"\nTesting basic stock fetch for {symbol}...")
    
    try:
        # Most basic possible call
        data = yf.download(
            symbol,
            start=datetime.now() - timedelta(days=5),
            end=datetime.now(),
            progress=False,
            show_errors=True
        )
        
        if not data.empty:
            print("\nSuccess! Got data:")
            print(data.tail())
        else:
            print("Error: Empty DataFrame returned")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_stock_fetch("AAPL") 