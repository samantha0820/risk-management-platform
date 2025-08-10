"""
Advanced Features Example
Demonstrates complex options and risk management functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.option_chain import OptionChainLoader
from models.monte_carlo import MonteCarloPricing
from risk.var_calculator import VaRCalculator
import pandas as pd
import numpy as np

def main():
    """Main function"""
    print("=== Options Pricing and Risk Management System - Advanced Features Example ===\n")
    
    # Initialize components
    data_loader = DataLoader()
    option_loader = OptionChainLoader()
    mc_model = MonteCarloPricing(n_simulations=5000)
    var_calculator = VaRCalculator()
    
    # 1. Complex options pricing
    print("1. Complex Options Pricing...")
    S0 = 100
    K = 100
    T = 0.5
    r = 0.05
    sigma = 0.2
    
    # Asian options
    asian_call = mc_model.price_asian_option(S0, K, T, r, sigma, 'call', 'arithmetic')
    asian_put = mc_model.price_asian_option(S0, K, T, r, sigma, 'put', 'arithmetic')
    
    print(f"   Asian Call Option (Arithmetic Average): ${asian_call['price']:.4f}")
    print(f"   Asian Put Option (Arithmetic Average): ${asian_put['price']:.4f}")
    
    # Barrier options
    barrier = 90
    barrier_call = mc_model.price_barrier_option(S0, K, T, r, sigma, barrier, 'call', 'down_and_out')
    barrier_put = mc_model.price_barrier_option(S0, K, T, r, sigma, barrier, 'put', 'down_and_out')
    
    print(f"   Barrier Call Option (Down and Out): ${barrier_call['price']:.4f}")
    print(f"   Barrier Put Option (Down and Out): ${barrier_put['price']:.4f}")
    
    # Lookback options
    lookback_call = mc_model.price_lookback_option(S0, T, r, sigma, 'call', 'floating')
    lookback_put = mc_model.price_lookback_option(S0, T, r, sigma, 'put', 'floating')
    
    print(f"   Lookback Call Option (Floating Strike): ${lookback_call['price']:.4f}")
    print(f"   Lookback Put Option (Floating Strike): ${lookback_put['price']:.4f}")
    
    # 2. Option chain analysis
    print("\n2. Option Chain Analysis...")
    symbol = 'AAPL'
    
    try:
        # Get option chain summary
        summary = option_loader.get_option_summary(symbol)
        print(f"   {symbol} Option Summary:")
        print(f"     Current Price: ${summary.get('current_price', 0):.2f}")
        print(f"     Available Expirations: {summary.get('available_expirations', 0)}")
        print(f"     Total Call Options: {summary.get('total_call_options', 0)}")
        print(f"     Total Put Options: {summary.get('total_put_options', 0)}")
        
        # Get at-the-money options
        atm_options = option_loader.get_atm_options(symbol)
        if not atm_options['calls'].empty:
            print(f"     At-the-money Call Options: {len(atm_options['calls'])}")
        if not atm_options['puts'].empty:
            print(f"     At-the-money Put Options: {len(atm_options['puts'])}")
            
    except Exception as e:
        print(f"   Unable to fetch option chain data: {e}")
    
    # 3. Portfolio risk analysis
    print("\n3. Portfolio Risk Analysis...")
    
    # Get multiple stock data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    portfolio_data = {}
    
    for sym in symbols:
        data = data_loader.get_stock_data(sym)
        if not data.empty:
            portfolio_data[sym] = data['Returns'].dropna()
    
    if len(portfolio_data) >= 2:
        # Create returns matrix
        returns_df = pd.DataFrame(portfolio_data)
        returns_df = returns_df.dropna()
        
        # Equal weight portfolio
        weights = [1/len(symbols)] * len(symbols)
        
        # Calculate portfolio VaR
        portfolio_var = var_calculator.portfolio_var(returns_df, weights, method='parametric')
        print(f"   Portfolio VaR (95%): {portfolio_var['var_annualized']:.2%}")
        
        # VaR decomposition
        var_decomp = var_calculator.var_decomposition(returns_df, weights)
        print("   VaR Decomposition:")
        for i, sym in enumerate(symbols):
            print(f"     {sym}: {var_decomp['contribution'][i]:.2%}")
    
    # 4. Stress testing
    print("\n4. Stress Testing...")
    
    # Get single stock data for stress testing
    stock_data = data_loader.get_stock_data('AAPL')
    if not stock_data.empty:
        returns = stock_data['Returns'].dropna()
        
        # Define stress scenarios
        stress_scenarios = {
            'Market Crash': 2.0,  # Double volatility
            'Moderate Decline': 1.5,  # 50% increase in volatility
            'Extreme Volatility': 3.0   # 200% increase in volatility
        }
        
        stress_results = var_calculator.stress_testing(returns, stress_scenarios)
        
        print("   Stress Testing Results:")
        for scenario, impact in stress_results.items():
            if 'impact' in scenario:
                print(f"     {scenario}: {impact:.2%}")
    
    # 5. Variance reduction techniques
    print("\n5. Variance Reduction Techniques...")
    
    # Use antithetic variates method
    antithetic_result = mc_model.variance_reduction_antithetic(S0, K, T, r, sigma, 'call')
    standard_result = mc_model.price_european_option(S0, K, T, r, sigma, 'call')
    
    print(f"   Standard Monte Carlo: ${standard_result['price']:.4f} ± {standard_result['std_error']:.4f}")
    print(f"   Antithetic Variates: ${antithetic_result['price']:.4f} ± {antithetic_result['std_error']:.4f}")
    
    # 6. Implied volatility analysis
    print("\n6. Implied Volatility Analysis...")
    
    # Simulate option prices for different strike prices
    strikes = np.linspace(80, 120, 9)
    implied_vols = []
    
    for strike in strikes:
        # Calculate theoretical price
        theoretical_price = mc_model.price_european_option(S0, strike, T, r, sigma, 'call')['price']
        
        # Back out implied volatility (simplified version)
        # In practice, more precise numerical methods should be used
        implied_vol = sigma  # Simplified here
        implied_vols.append(implied_vol)
    
    print("   Implied Volatility Curve (Simplified):")
    for i, strike in enumerate(strikes):
        print(f"     Strike ${strike:.0f}: {implied_vols[i]:.3f}")
    
    # 7. Risk metrics comparison
    print("\n7. Risk Metrics Comparison...")
    
    if not stock_data.empty:
        returns = stock_data['Returns'].dropna()
        
        # Calculate multiple risk metrics
        var_95 = var_calculator.parametric_var(returns, 0.95)
        var_99 = var_calculator.parametric_var(returns, 0.99)
        cvar_95 = var_calculator.parametric_var(returns, 0.95)
        
        print(f"   VaR (95%): {var_95['var_annualized']:.2%}")
        print(f"   VaR (99%): {var_99['var_annualized']:.2%}")
        print(f"   CVaR (95%): {cvar_95['cvar_annualized']:.2%}")
        
        # Calculate other risk metrics
        volatility = returns.std() * np.sqrt(252)
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        
        print(f"   Annualized Volatility: {volatility:.2%}")
        print(f"   Maximum Drawdown: {max_drawdown:.2%}")
    
    print("\n=== Advanced Features Example Complete ===")

if __name__ == "__main__":
    main()
