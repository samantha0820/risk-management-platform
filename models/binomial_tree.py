"""
Binomial Tree Options Pricing Model
Supports European and American options pricing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import config


class BinomialTree:
    """Binomial Tree Options Pricing Model"""
    
    def __init__(self, steps: int = config.BINOMIAL_STEPS):
        self.steps = steps
    
    def _calculate_parameters(self, S: float, K: float, T: float, r: float, sigma: float) -> Dict:
        """
        Calculate binomial tree parameters
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Dictionary containing tree parameters
        """
        dt = T / self.steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        return {
            'dt': dt,
            'u': u,
            'd': d,
            'p': p,
            'discount': np.exp(-r * dt)
        }
    
    def _build_price_tree(self, S: float, params: Dict) -> np.ndarray:
        """
        Build stock price tree
        
        Args:
            S: Initial stock price
            params: Tree parameters
        
        Returns:
            Stock price tree matrix
        """
        price_tree = np.zeros((self.steps + 1, self.steps + 1))
        
        for i in range(self.steps + 1):
            for j in range(i + 1):
                price_tree[i, j] = S * (params['u'] ** (i - j)) * (params['d'] ** j)
        
        return price_tree
    
    def _calculate_payoff(self, price_tree: np.ndarray, K: float, option_type: str) -> np.ndarray:
        """
        Calculate payoff at expiry
        
        Args:
            price_tree: Stock price tree
            K: Strike price
            option_type: Option type
        
        Returns:
            Payoff tree
        """
        payoff_tree = np.zeros_like(price_tree)
        
        if option_type.lower() == 'call':
            payoff_tree[self.steps, :] = np.maximum(price_tree[self.steps, :] - K, 0)
        else:  # put
            payoff_tree[self.steps, :] = np.maximum(K - price_tree[self.steps, :], 0)
        
        return payoff_tree
    
    def _backward_induction(self, payoff_tree: np.ndarray, price_tree: np.ndarray, 
                           params: Dict, option_type: str, american: bool = False, 
                           strike: float = None) -> np.ndarray:
        """
        Backward induction to calculate option prices
        
        Args:
            payoff_tree: Payoff tree
            price_tree: Stock price tree
            params: Tree parameters
            option_type: Option type
            american: Whether it's an American option
            strike: Strike price
        
        Returns:
            Option price tree
        """
        option_tree = payoff_tree.copy()
        
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # Calculate risk-neutral expected value
                expected_value = (params['p'] * option_tree[i + 1, j] + 
                                (1 - params['p']) * option_tree[i + 1, j + 1])
                option_tree[i, j] = params['discount'] * expected_value
                
                # American option early exercise check
                if american and strike is not None:
                    if option_type.lower() == 'call':
                        exercise_value = max(price_tree[i, j] - strike, 0)
                    else:  # put
                        exercise_value = max(strike - price_tree[i, j], 0)
                    
                    option_tree[i, j] = max(option_tree[i, j], exercise_value)
        
        return option_tree
    
    def price_option(self, S: float, K: float, T: float, r: float, sigma: float, 
                    option_type: str = 'call', american: bool = False) -> Dict:
        """
        Calculate option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type ('call' or 'put')
            american: Whether it's an American option
        
        Returns:
            Dictionary containing price and trees
        """
        if T <= 0:
            if option_type.lower() == 'call':
                price = max(S - K, 0)
            else:
                price = max(K - S, 0)
            return {'price': price, 'price_tree': None, 'option_tree': None}
        
        # Calculate parameters
        params = self._calculate_parameters(S, K, T, r, sigma)
        
        # Build stock price tree
        price_tree = self._build_price_tree(S, params)
        
        # Calculate payoff
        payoff_tree = self._calculate_payoff(price_tree, K, option_type)
        
        # Backward induction
        option_tree = self._backward_induction(payoff_tree, price_tree, params, option_type, american, K)
        
        return {
            'price': option_tree[0, 0],
            'price_tree': price_tree,
            'option_tree': option_tree,
            'params': params
        }
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str = 'call', american: bool = False, 
                        dS: float = 0.01, dsigma: float = 0.01, dr: float = 0.01) -> Dict:
        """
        Calculate Greeks (using finite difference method)
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type
            american: Whether it's an American option
            dS: Stock price change
            dsigma: Volatility change
            dr: Interest rate change
        
        Returns:
            Dictionary containing Greeks
        """
        # Calculate base price
        base_result = self.price_option(S, K, T, r, sigma, option_type, american)
        base_price = base_result['price']
        
        # Delta
        price_up = self.price_option(S + dS, K, T, r, sigma, option_type, american)['price']
        price_down = self.price_option(S - dS, K, T, r, sigma, option_type, american)['price']
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma
        gamma = (price_up + price_down - 2 * base_price) / (dS ** 2)
        
        # Vega
        price_vol_up = self.price_option(S, K, T, r, sigma + dsigma, option_type, american)['price']
        price_vol_down = self.price_option(S, K, T, r, sigma - dsigma, option_type, american)['price']
        vega = (price_vol_up - price_vol_down) / (2 * dsigma)
        
        # Rho
        price_r_up = self.price_option(S, K, T, r + dr, sigma, option_type, american)['price']
        price_r_down = self.price_option(S, K, T, r - dr, sigma, option_type, american)['price']
        rho = (price_r_up - price_r_down) / (2 * dr)
        
        # Theta (using time change)
        dt = 0.01
        price_time = self.price_option(S, K, T - dt, r, sigma, option_type, american)['price']
        theta = (price_time - base_price) / dt
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def price_american_option(self, S: float, K: float, T: float, r: float, sigma: float, 
                            option_type: str = 'call') -> Dict:
        """
        Calculate American option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type
        
        Returns:
            Dictionary containing price and tree
        """
        return self.price_option(S, K, T, r, sigma, option_type, american=True)
    
    def compare_european_american(self, S: float, K: float, T: float, r: float, sigma: float, 
                                option_type: str = 'call') -> Dict:
        """
        Compare European and American option prices
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type
        
        Returns:
            Comparison result dictionary
        """
        european = self.price_option(S, K, T, r, sigma, option_type, american=False)
        american = self.price_option(S, K, T, r, sigma, option_type, american=True)
        
        return {
            'european_price': european['price'],
            'american_price': american['price'],
            'early_exercise_premium': american['price'] - european['price'],
            'european_tree': european,
            'american_tree': american
        }
    
    def sensitivity_analysis(self, S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str = 'call', american: bool = False) -> Dict:
        """
        Sensitivity analysis
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type
            american: Whether it's an American option
        
        Returns:
            Sensitivity analysis results
        """
        # Stock price sensitivity
        S_range = np.linspace(S * 0.5, S * 1.5, 20)
        prices_S = []
        for s in S_range:
            price = self.price_option(s, K, T, r, sigma, option_type, american)['price']
            prices_S.append(price)
        
        # Volatility sensitivity
        sigma_range = np.linspace(0.1, 0.5, 20)
        prices_sigma = []
        for vol in sigma_range:
            price = self.price_option(S, K, T, r, vol, option_type, american)['price']
            prices_sigma.append(price)
        
        # Time sensitivity
        T_range = np.linspace(0.1, T * 2, 20)
        prices_T = []
        for t in T_range:
            price = self.price_option(S, K, t, r, sigma, option_type, american)['price']
            prices_T.append(price)
        
        return {
            'stock_prices': S_range,
            'option_prices_stock': prices_S,
            'volatilities': sigma_range,
            'option_prices_vol': prices_sigma,
            'times': T_range,
            'option_prices_time': prices_T
        }
