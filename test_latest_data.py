"""
Test fetching latest data
"""

from data.data_loader import DataLoader
from datetime import datetime

def test_latest_data():
    """Test fetching latest data"""
    print("Testing latest stock data retrieval...")
    
    data_loader = DataLoader()
    
    # Test TSLA data
    print("\nFetching TSLA data...")
    tsla_data = data_loader.get_stock_data('TSLA', use_cache=False)
    
    if not tsla_data.empty:
        print(f"Data range: {tsla_data.index[0].strftime('%Y-%m-%d')} to {tsla_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Data points: {len(tsla_data)}")
        print(f"Latest price: ${tsla_data['Close'].iloc[-1]:.2f}")
        print(f"Latest date: {tsla_data.index[-1].strftime('%Y-%m-%d')}")
        
        # Check if 2024 data is included
        data_2024 = tsla_data[tsla_data.index.year >= 2024]
        if not data_2024.empty:
            print(f"✅ Contains 2024 data: {len(data_2024)} data points")
        else:
            print("❌ No 2024 data")
    else:
        print("❌ Unable to fetch TSLA data")
    
    # Test AAPL data
    print("\nFetching AAPL data...")
    aapl_data = data_loader.get_stock_data('AAPL', use_cache=False)
    
    if not aapl_data.empty:
        print(f"Data range: {aapl_data.index[0].strftime('%Y-%m-%d')} to {aapl_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Data points: {len(aapl_data)}")
        print(f"Latest price: ${aapl_data['Close'].iloc[-1]:.2f}")
        print(f"Latest date: {aapl_data.index[-1].strftime('%Y-%m-%d')}")
        
        # Check if 2024 data is included
        data_2024 = aapl_data[aapl_data.index.year >= 2024]
        if not data_2024.empty:
            print(f"✅ Contains 2024 data: {len(data_2024)} data points")
        else:
            print("❌ No 2024 data")

if __name__ == "__main__":
    test_latest_data()
