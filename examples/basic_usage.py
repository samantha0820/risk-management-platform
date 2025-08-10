"""
Basic Usage Example
Demonstrates how to use the Options Pricing and Risk Management System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from models.black_scholes import BlackScholes
from models.binomial_tree import BinomialTree
from models.monte_carlo import MonteCarloPricing
from risk.var_calculator import VaRCalculator
import config

def main():
    """Main function"""
    print("=== Options Pricing and Risk Management System - Basic Usage Example ===\n")
    
    # Initialize components
    data_loader = DataLoader()
    bs_model = BlackScholes()
    binomial_model = BinomialTree()
    mc_model = MonteCarloPricing()
    var_calculator = VaRCalculator()
    
    # 1. Get stock data
    print("1. Getting Stock Data...")
    symbol = 'AAPL'
    stock_data = data_loader.get_stock_data(symbol)
    
    if not stock_data.empty:
        current_price = stock_data['Close'].iloc[-1]
        print(f"   {symbol} Current Price: ${current_price:.2f}")
        print(f"   Historical Data Points: {len(stock_data)}")
    else:
        print(f"   Unable to fetch {symbol} data")
        return
    
    # 2. Calculate option prices
    print("\n2. Calculating Option Prices...")
    S = current_price
    K = 150  # Strike price
    T = 0.25  # 3 months
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    
    # Black-Scholes
    call_price_bs = bs_model.price_call(S, K, T, r, sigma)
    put_price_bs = bs_model.price_put(S, K, T, r, sigma)
    
    # Binomial Tree
    call_price_bin = binomial_model.price_option(S, K, T, r, sigma, 'call')['price']
    put_price_bin = binomial_model.price_option(S, K, T, r, sigma, 'put')['price']
    
    # Monte Carlo
    call_price_mc = mc_model.price_european_option(S, K, T, r, sigma, 'call')['price']
    put_price_mc = mc_model.price_european_option(S, K, T, r, sigma, 'put')['price']
    
    print(f"   Call Option Prices:")
    print(f"     Black-Scholes: ${call_price_bs:.4f}")
    print(f"     Binomial Tree: ${call_price_bin:.4f}")
    print(f"     Monte Carlo: ${call_price_mc:.4f}")
    
    print(f"   Put Option Prices:")
    print(f"     Black-Scholes: ${put_price_bs:.4f}")
    print(f"     Binomial Tree: ${put_price_bin:.4f}")
    print(f"     Monte Carlo: ${put_price_mc:.4f}")
    
    # 3. Calculate Greeks
    print("\n3. Calculating Greeks...")
    greeks_call = bs_model.calculate_all_greeks('call', S, K, T, r, sigma)
    greeks_put = bs_model.calculate_all_greeks('put', S, K, T, r, sigma)
    
    print(f"   Call Option Greeks:")
    for greek, value in greeks_call.items():
        if greek != 'price':
            print(f"     {greek.capitalize()}: {value:.4f}")
    
    print(f"   Put Option Greeks:")
    for greek, value in greeks_put.items():
        if greek != 'price':
            print(f"     {greek.capitalize()}: {value:.4f}")
    
    # 4. Risk analysis
    print("\n4. Risk Analysis...")
    returns = stock_data['Returns'].dropna()
    
    # Calculate VaR
    var_result = var_calculator.parametric_var(returns)
    print(f"   Parametric VaR (95%): {var_result['var_annualized']:.2%}")
    
    # Historical VaR
    hist_var_result = var_calculator.historical_var(returns)
    print(f"   Historical VaR (95%): {hist_var_result['var_annualized']:.2%}")
    
    # Monte Carlo VaR
    mc_var_result = var_calculator.monte_carlo_var(returns)
    print(f"   Monte Carlo VaR (95%): {mc_var_result['var_annualized']:.2%}")
    
    # 5. Sensitivity analysis
    print("\n5. Sensitivity Analysis...")
    
    # Stock price sensitivity
    S_range = [S * 0.8, S * 0.9, S, S * 1.1, S * 1.2]
    print("   Stock Price Sensitivity (Call Option):")
    for s in S_range:
        price = bs_model.price_call(s, K, T, r, sigma)
        print(f"     Stock Price ${s:.2f}: ${price:.4f}")
    
    # Volatility sensitivity
    sigma_range = [0.1, 0.15, 0.2, 0.25, 0.3]
    print("   Volatility Sensitivity (Call Option):")
    for vol in sigma_range:
        price = bs_model.price_call(S, K, T, r, vol)
        print(f"     Volatility {vol:.2f}: ${price:.4f}")
    
    # 6. American option comparison
    print("\n6. American Option Comparison...")
    american_call = binomial_model.price_american_option(S, K, T, r, sigma, 'call')
    american_put = binomial_model.price_american_option(S, K, T, r, sigma, 'put')
    
    print(f"   American Call Option: ${american_call['price']:.4f}")
    print(f"   American Put Option: ${american_put['price']:.4f}")
    print(f"   Early Exercise Premium (Call): ${american_call['price'] - call_price_bs:.4f}")
    print(f"   Early Exercise Premium (Put): ${american_put['price'] - put_price_bs:.4f}")
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main()
