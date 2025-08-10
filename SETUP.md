# Setup Guide

## 📋 System Requirements

- Python 3.8+
- Internet connection (for fetching market data)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Demo
```bash
python run_demo.py
```

### 3. Start Dashboard
```bash
python app.py
```

## 🔑 API Setup (Optional)

### Yahoo Finance API
- **No account required** ✅
- System automatically uses yfinance package to fetch data
- Completely free, no setup required

### FRED API (Optional)
- **Free account required** ⚠️
- Used for fetching risk-free rate data
- If not set up, system will use default values

#### How to get FRED API key:

1. Visit [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Register for a free account
3. Get API key
4. Set environment variable:

```bash
# macOS/Linux
export FRED_API_KEY="your_api_key_here"

# Windows
set FRED_API_KEY=your_api_key_here
```

Or set in Python:
```python
import os
os.environ['FRED_API_KEY'] = 'your_api_key_here'
```

## 📊 Data Sources

### Stock Data
- **Source**: Yahoo Finance
- **Cost**: Free
- **Limitations**: Reasonable request frequency
- **Data**: Historical prices, option chains, fundamentals

### Interest Rate Data
- **Source**: FRED (Federal Reserve)
- **Cost**: Free
- **Limitations**: Reasonable API call limits
- **Data**: Treasury yields, interest rate indicators

## 🔧 Troubleshooting

### Cannot fetch stock data
- Check internet connection
- Verify stock symbol is correct
- Wait a while and retry (may hit frequency limits)

### Cannot fetch option data
- Some stocks may not have option trading
- Try other stock symbols (like AAPL, MSFT, SPY)

### FRED API errors
- Check if API key is correct
- Verify account status
- If not set up, system will use default interest rates

## 📈 Recommended Stock Symbols

### High Liquidity Stocks
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Google)
- AMZN (Amazon)
- TSLA (Tesla)

### Index ETFs
- SPY (S&P 500)
- QQQ (NASDAQ 100)
- IWM (Russell 2000)

### Volatility Index
- VIX (Fear Index)

## 🎯 Usage Recommendations

1. **First time use**: Run `python run_demo.py` to see system features
2. **Interactive analysis**: Use `python app.py` to start dashboard
3. **Learning examples**: Check example code in `examples/` directory
4. **Test functionality**: Run `python -m unittest tests/test_models.py`

## 📞 Support

If you encounter issues:
1. Check internet connection
2. Verify Python version (3.8+)
3. Reinstall dependencies: `pip install -r requirements.txt`
4. Check error messages and refer to troubleshooting section
