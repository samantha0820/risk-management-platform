# Options Pricing and Risk Management System

## Project Overview
This is a comprehensive options pricing and risk management system that implements multiple pricing models, Greeks calculations, and risk management tools.

## Key Features
- **Pricing Models**: Black-Scholes, Binomial Tree, Monte Carlo Simulation
- **Greeks Calculations**: Delta, Gamma, Vega, Theta, Rho
- **Risk Management**: VaR, CVaR, Scenario Analysis
- **Data Sources**: Yahoo Finance, FRED API
- **Visualization**: Interactive Charts and Dashboard

## Installation and Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
Set your FRED API key in `config.py` (optional)

### 3. Run Application
```bash
python app.py
```

## Project Structure
```
├── data/                   # Data processing modules
├── models/                 # Pricing models
├── risk/                   # Risk management
├── utils/                  # Utility functions
├── dashboard/              # Dashboard
├── tests/                  # Test files
├── config.py              # Configuration
├── app.py                 # Main application
└── requirements.txt       # Dependencies
```

## Usage Examples
```python
from models.black_scholes import BlackScholes
from data.data_loader import DataLoader

# Load data
loader = DataLoader()
stock_data = loader.get_stock_data('AAPL')

# Calculate option price
bs = BlackScholes()
price = bs.price_call(S=100, K=100, T=1, r=0.05, sigma=0.2)
```
