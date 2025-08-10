"""
Options Pricing and Risk Management System Demo Script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import DataLoader
from models.black_scholes import BlackScholes
from models.binomial_tree import BinomialTree
from models.monte_carlo import MonteCarloPricing
from risk.var_calculator import VaRCalculator
import config

def main():
    """Main demo function"""
    print("=" * 60)
    print("Options Pricing and Risk Management System Demo")
    print("=" * 60)
    
    # Initialize components
    print("\nInitializing system components...")
    data_loader = DataLoader()
    bs_model = BlackScholes()
    binomial_model = BinomialTree()
    mc_model = MonteCarloPricing()
    var_calculator = VaRCalculator()
    
    # Demo 1: Basic options pricing
    print("\n1. Basic Options Pricing Demo")
    print("-" * 40)
    
    # Get AAPL data
    print("Fetching AAPL stock data...")
    stock_data = data_loader.get_stock_data('AAPL')
    
    if not stock_data.empty:
        current_price = stock_data['Close'].iloc[-1]
        print(f"Current AAPL price: ${current_price:.2f}")
        
        # Calculate option prices
        S = current_price
        K = current_price  # At-the-money option
        T = 0.25  # 3 months
        r = 0.05  # 5% risk-free rate
        sigma = 0.2  # 20% volatility
        
        print(f"\nOption Parameters:")
        print(f"  Underlying Price: ${S:.2f}")
        print(f"  Strike Price: ${K:.2f}")
        print(f"  Time to Expiry: {T:.2f} years")
        print(f"  Risk-free Rate: {r:.1%}")
        print(f"  Volatility: {sigma:.1%}")
        
        # Calculate call option prices
        call_bs = bs_model.price_call(S, K, T, r, sigma)
        call_bin = binomial_model.price_option(S, K, T, r, sigma, 'call')['price']
        call_mc = mc_model.price_european_option(S, K, T, r, sigma, 'call')['price']
        
        print(f"\nCall Option Prices:")
        print(f"  Black-Scholes: ${call_bs:.4f}")
        print(f"  Binomial Tree: ${call_bin:.4f}")
        print(f"  Monte Carlo: ${call_mc:.4f}")
        
        # Calculate put option prices
        put_bs = bs_model.price_put(S, K, T, r, sigma)
        put_bin = binomial_model.price_option(S, K, T, r, sigma, 'put')['price']
        put_mc = mc_model.price_european_option(S, K, T, r, sigma, 'put')['price']
        
        print(f"\nPut Option Prices:")
        print(f"  Black-Scholes: ${put_bs:.4f}")
        print(f"  Binomial Tree: ${put_bin:.4f}")
        print(f"  Monte Carlo: ${put_mc:.4f}")
        
        # Verify put-call parity
        parity_left = call_bs - put_bs
        parity_right = S - K * (1 / (1 + r) ** T)
        print(f"\nPut-Call Parity Verification:")
        print(f"  C - P = ${parity_left:.4f}")
        print(f"  S - K*exp(-rT) = ${parity_right:.4f}")
        print(f"  Difference: ${abs(parity_left - parity_right):.6f}")
        
    else:
        print("Unable to fetch stock data")
        return
    
    # Demo 2: Greeks calculation
    print("\n\n2. Greeks Calculation Demo")
    print("-" * 40)
    
    greeks_call = bs_model.calculate_all_greeks('call', S, K, T, r, sigma)
    greeks_put = bs_model.calculate_all_greeks('put', S, K, T, r, sigma)
    
    print("Call Option Greeks:")
    for greek, value in greeks_call.items():
        if greek != 'price':
            print(f"  {greek.capitalize()}: {value:.4f}")
    
    print("\nPut Option Greeks:")
    for greek, value in greeks_put.items():
        if greek != 'price':
            print(f"  {greek.capitalize()}: {value:.4f}")
    
    # Demo 3: Risk analysis
    print("\n\n3. Risk Analysis Demo")
    print("-" * 40)
    
    returns = stock_data['Returns'].dropna()
    
    # Calculate VaR
    var_95 = var_calculator.parametric_var(returns, 0.95)
    var_99 = var_calculator.parametric_var(returns, 0.99)
    
    print(f"VaR Analysis (based on AAPL historical data):")
    print(f"  95% VaR: {var_95['var_annualized']:.2%}")
    print(f"  99% VaR: {var_99['var_annualized']:.2%}")
    print(f"  95% CVaR: {var_95['cvar_annualized']:.2%}")
    
    # Calculate other risk metrics
    volatility = returns.std() * (252 ** 0.5)
    print(f"  Annualized Volatility: {volatility:.2%}")
    
    # Demo 4: Sensitivity analysis
    print("\n\n4. Sensitivity Analysis Demo")
    print("-" * 40)
    
    # Stock price sensitivity
    S_range = [S * 0.8, S * 0.9, S, S * 1.1, S * 1.2]
    print("Stock Price Impact on Call Option Price:")
    for s in S_range:
        price = bs_model.price_call(s, K, T, r, sigma)
        print(f"  Stock Price ${s:.2f}: ${price:.4f}")
    
    # Volatility sensitivity
    sigma_range = [0.1, 0.15, 0.2, 0.25, 0.3]
    print("\nVolatility Impact on Call Option Price:")
    for vol in sigma_range:
        price = bs_model.price_call(S, K, T, r, vol)
        print(f"  Volatility {vol:.1%}: ${price:.4f}")
    
    # Demo 5: Complex options
    print("\n\n5. Complex Options Demo")
    print("-" * 40)
    
    # Asian options
    asian_call = mc_model.price_asian_option(S, K, T, r, sigma, 'call', 'arithmetic')
    print(f"Asian Call Option (Arithmetic Average): ${asian_call['price']:.4f}")
    
    # Barrier options
    barrier = S * 0.9  # 90% of current price
    barrier_call = mc_model.price_barrier_option(S, K, T, r, sigma, barrier, 'call', 'down_and_out')
    print(f"Barrier Call Option (Down and Out): ${barrier_call['price']:.4f}")
    
    # Lookback options
    lookback_call = mc_model.price_lookback_option(S, T, r, sigma, 'call', 'floating')
    print(f"Lookback Call Option (Floating Strike): ${lookback_call['price']:.4f}")
    
    # Demo 6: American options
    print("\n\n6. American Options Demo")
    print("-" * 40)
    
    american_call = binomial_model.price_american_option(S, K, T, r, sigma, 'call')
    american_put = binomial_model.price_american_option(S, K, T, r, sigma, 'put')
    
    print(f"American Call Option: ${american_call['price']:.4f}")
    print(f"American Put Option: ${american_put['price']:.4f}")
    print(f"Early Exercise Premium (Call): ${american_call['price'] - call_bs:.4f}")
    print(f"Early Exercise Premium (Put): ${american_put['price'] - put_bs:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nTo start the interactive dashboard, run: python app.py")
    print("To view basic usage examples, run: python examples/basic_usage.py")
    print("To view advanced features examples, run: python examples/advanced_features.py")

if __name__ == "__main__":
    main()
