from utils import get_stock_data
import os
from dotenv import load_dotenv
import pandas as pd
import requests
import json

def test_alpha_vantage():
    print("\nTesting stock fetch for AAPL...")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    print(f"API Key found: {'Yes' if api_key else 'No'}")
    print(f"API Key length: {len(api_key) if api_key else 'N/A'}")
    
    try:
        # Make a direct API request first
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={api_key}&outputsize=compact'
        response = requests.get(url, timeout=10)
        
        print("\nResponse details:")
        print(f"Status code: {response.status_code}")
        print(f"Content type: {response.headers.get('content-type')}")
        print(f"Response length: {len(response.text)}")
        
        # Try to parse JSON directly
        print("\nTrying direct JSON parsing...")
        try:
            cleaned_text = response.text.strip()
            print(f"First 100 chars of cleaned text: {cleaned_text[:100]}")
            data = json.loads(cleaned_text)
            print("Direct JSON parsing successful!")
            print("Available keys:", list(data.keys()))
        except json.JSONDecodeError as e:
            print(f"Direct JSON parsing failed: {e}")
            print("Response content:", response.text[:500])
        
        # Now try the full function
        print("\nTrying full get_stock_data function...")
        df = get_stock_data('AAPL', period='1mo')
        
        if not df.empty:
            print("\nSuccessfully fetched data!")
            print("\nFirst 5 rows:")
            print(df.head())
            print("\nShape:", df.shape)
            print("\nDate range:", df.index.min(), "to", df.index.max())
            print("\nSummary statistics:")
            print(df.describe())
        else:
            print("Error: Empty DataFrame returned")
            
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    test_alpha_vantage() 