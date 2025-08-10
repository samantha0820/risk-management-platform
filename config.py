"""
Configuration file
Contains API keys, basic parameters and settings
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List

# API Configuration
FRED_API_KEY = os.getenv('FRED_API_KEY', '')  # Read FRED API key from environment variables
# If no FRED API key is set, the system will use default values

# Stock symbols configuration
STOCK_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ']

# Options configuration
OPTION_TYPES = ['call', 'put']
DEFAULT_RISK_FREE_RATE = 0.05
DEFAULT_VOLATILITY = 0.2

# Data configuration
DEFAULT_START_DATE = '2020-01-01'
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')  # Automatically get current date
DATA_CACHE_DIR = 'data/cache'

# Model configuration
BINOMIAL_STEPS = 100
MONTE_CARLO_SIMULATIONS = 10000
MONTE_CARLO_STEPS = 252

# Risk management configuration
VAR_CONFIDENCE_LEVEL = 0.95
HISTORICAL_DAYS = 252

# Visualization configuration
PLOTLY_TEMPLATE = 'plotly_white'
FIGURE_HEIGHT = 600
FIGURE_WIDTH = 800

# Color configuration
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8'
}

# Options chain configuration
OPTION_CHAIN_CONFIG = {
    'max_days_to_expiry': 365,
    'min_volume': 10,
    'min_open_interest': 50
}
