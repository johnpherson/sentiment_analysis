from utils import get_stock_data
import pandas as pd
from datetime import datetime, timedelta

def test_stock_data():
    """Test the updated get_stock_data function with different scenarios"""
    
    # Test case 1: Basic stock fetch
    print("\nTest 1: Basic stock fetch (AAPL)")
    try:
        data = get_stock_data('AAPL', period='7d')
        print("✓ Successfully fetched AAPL data")
        print(f"Shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print("\nFirst few rows:")
        print(data.head())
    except Exception as e:
        print(f"✗ Error fetching AAPL data: {str(e)}")

    # Test case 2: Test caching
    print("\nTest 2: Testing cache functionality")
    try:
        print("First fetch (should hit API):")
        data1 = get_stock_data('MSFT', period='7d')
        print("\nSecond fetch (should use cache):")
        data2 = get_stock_data('MSFT', period='7d')
        print("✓ Cache test completed")
    except Exception as e:
        print(f"✗ Error during cache test: {str(e)}")

    # Test case 3: Invalid symbol
    print("\nTest 3: Testing invalid symbol handling")
    try:
        data = get_stock_data('INVALID_SYMBOL_123', period='7d')
        print("Note: Should have fallen back to dummy data")
        print(f"Shape: {data.shape}")
    except Exception as e:
        print(f"✗ Unexpected error with invalid symbol: {str(e)}")

    # Test case 4: Different period format
    print("\nTest 4: Testing different period format")
    try:
        data = get_stock_data('GOOGL', period='1mo')
        print("✓ Successfully fetched GOOGL monthly data")
        print(f"Shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
    except Exception as e:
        print(f"✗ Error with monthly period: {str(e)}")

if __name__ == "__main__":
    test_stock_data() 