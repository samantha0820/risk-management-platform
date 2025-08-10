"""
Test dynamic date configuration
"""

import config
from datetime import datetime

def test_dynamic_date():
    """Test dynamic date configuration"""
    print("Testing dynamic date configuration...")
    
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"End date in config: {config.DEFAULT_END_DATE}")
    print(f"Start date in config: {config.DEFAULT_START_DATE}")
    
    # Check if date format is correct
    try:
        datetime.strptime(config.DEFAULT_END_DATE, '%Y-%m-%d')
        print("✅ Date format is correct")
    except ValueError:
        print("❌ Date format is incorrect")
    
    # Check if dates are reasonable
    start_date = datetime.strptime(config.DEFAULT_START_DATE, '%Y-%m-%d')
    end_date = datetime.strptime(config.DEFAULT_END_DATE, '%Y-%m-%d')
    
    if end_date > start_date:
        print("✅ End date is after start date")
    else:
        print("❌ End date is before start date")
    
    # Calculate data range
    date_range = (end_date - start_date).days
    print(f"Data range: {date_range} days")

if __name__ == "__main__":
    test_dynamic_date()
