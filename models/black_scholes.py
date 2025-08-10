"""
Black-Scholes Options Pricing Model
Implements European options pricing and Greeks calculation
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Tuple, Optional
import config


class BlackScholes:
    """Black-Scholes Options Pricing Model"""
    
    def __init__(self):
        pass
    
    def _d1_d2(self, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            (d1, d2) tuple
        """
        if T <= 0:
            return (0, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        return d1, d2
    
    def price_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate call option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Call option price
        """
        if T <= 0:
            return max(S - K, 0)
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price
    
    def price_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate put option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Put option price
        """
        if T <= 0:
            return max(K - S, 0)
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put_price
    
    def delta_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate call option Delta
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Delta value
        """
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        return norm.cdf(d1)
    
    def delta_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate put option Delta
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Delta value
        """
        if T <= 0:
            return -1.0 if S < K else 0.0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        return norm.cdf(d1) - 1
    
    def gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Gamma (same for call and put)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Gamma value
        """
        if T <= 0:
            return 0.0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    def vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Vega (same for call and put)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Vega value
        """
        if T <= 0:
            return 0.0
        
        d1, _ = self._d1_d2(S, K, T, r, sigma)
        return S * np.sqrt(T) * norm.pdf(d1)
    
    def theta_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate call option Theta
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Theta value
        """
        if T <= 0:
            return 0.0
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                r * K * np.exp(-r * T) * norm.cdf(d2))
        return theta
    
    def theta_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate put option Theta
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Theta value
        """
        if T <= 0:
            return 0.0
        
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                r * K * np.exp(-r * T) * norm.cdf(-d2))
        return theta
    
    def rho_call(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate call option Rho
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Rho value
        """
        if T <= 0:
            return 0.0
        
        _, d2 = self._d1_d2(S, K, T, r, sigma)
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    
    def rho_put(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate put option Rho
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Rho value
        """
        if T <= 0:
            return 0.0
        
        _, d2 = self._d1_d2(S, K, T, r, sigma)
        return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    def calculate_all_greeks(self, option_type: str, S: float, K: float, T: float, 
                           r: float, sigma: float) -> Dict[str, float]:
        """
        Calculate all Greeks
        
        Args:
            option_type: Option type ('call' or 'put')
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Dictionary containing all Greeks
        """
        if option_type.lower() == 'call':
            return {
                'price': self.price_call(S, K, T, r, sigma),
                'delta': self.delta_call(S, K, T, r, sigma),
                'gamma': self.gamma(S, K, T, r, sigma),
                'vega': self.vega(S, K, T, r, sigma),
                'theta': self.theta_call(S, K, T, r, sigma),
                'rho': self.rho_call(S, K, T, r, sigma)
            }
        elif option_type.lower() == 'put':
            return {
                'price': self.price_put(S, K, T, r, sigma),
                'delta': self.delta_put(S, K, T, r, sigma),
                'gamma': self.gamma(S, K, T, r, sigma),
                'vega': self.vega(S, K, T, r, sigma),
                'theta': self.theta_put(S, K, T, r, sigma),
                'rho': self.rho_put(S, K, T, r, sigma)
            }
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def implied_volatility(self, option_type: str, market_price: float, S: float, 
                          K: float, T: float, r: float, tolerance: float = 1e-5, 
                          max_iterations: int = 100) -> float:
        """
        Calculate implied volatility
        
        Args:
            option_type: Option type ('call' or 'put')
            market_price: Market price
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
        
        Returns:
            Implied volatility
        """
        if T <= 0:
            return 0.0
        
        # Use Newton's method to solve for implied volatility
        sigma = 0.5  # Initial guess
        
        for i in range(max_iterations):
            if option_type.lower() == 'call':
                price = self.price_call(S, K, T, r, sigma)
                vega = self.vega(S, K, T, r, sigma)
            else:
                price = self.price_put(S, K, T, r, sigma)
                vega = self.vega(S, K, T, r, sigma)
            
            diff = market_price - price
            
            if abs(diff) < tolerance:
                return sigma
            
            if abs(vega) < 1e-10:
                break
            
            sigma = sigma + diff / vega
            sigma = max(0.001, min(5.0, sigma))  # Limit to reasonable range
        
        return sigma
    
    def price_option_chain(self, option_chain: pd.DataFrame, S: float, T: float, 
                          r: float, sigma: float) -> pd.DataFrame:
        """
        Calculate theoretical prices and Greeks for entire option chain
        
        Args:
            option_chain: Option chain DataFrame
            S: Current stock price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            DataFrame containing theoretical prices and Greeks
        """
        if option_chain.empty:
            return option_chain
        
        result_df = option_chain.copy()
        
        for idx, row in result_df.iterrows():
            K = row['strike']
            option_type = 'call' if 'C' in str(row.get('contractSymbol', '')) else 'put'
            
            greeks = self.calculate_all_greeks(option_type, S, K, T, r, sigma)
            
            result_df.at[idx, 'theoretical_price'] = greeks['price']
            result_df.at[idx, 'delta'] = greeks['delta']
            result_df.at[idx, 'gamma'] = greeks['gamma']
            result_df.at[idx, 'vega'] = greeks['vega']
            result_df.at[idx, 'theta'] = greeks['theta']
            result_df.at[idx, 'rho'] = greeks['rho']
        
        return result_df
