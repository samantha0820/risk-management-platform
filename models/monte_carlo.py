"""
Monte Carlo Options Pricing Model
Supports multiple stochastic processes and complex option structures
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from scipy.stats import norm
import config


class MonteCarloPricing:
    """Monte Carlo Options Pricing Model"""
    
    def __init__(self, n_simulations: int = config.MONTE_CARLO_SIMULATIONS, 
                 n_steps: int = config.MONTE_CARLO_STEPS):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
    
    def _generate_gbm_paths(self, S0: float, T: float, r: float, sigma: float) -> np.ndarray:
        """
        Generate Geometric Brownian Motion paths
        
        Args:
            S0: Initial stock price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
        
        Returns:
            Stock price path matrix (n_simulations x n_steps)
        """
        dt = T / self.n_steps
        
        # Generate random numbers
        Z = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        
        # Calculate paths
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Cumulative log returns
        log_returns = drift + diffusion * Z
        cumulative_returns = np.cumsum(log_returns, axis=1)
        
        # Calculate stock price paths
        paths = S0 * np.exp(cumulative_returns)
        
        return paths
    
    def _generate_heston_paths(self, S0: float, T: float, r: float, sigma0: float,
                              kappa: float, theta: float, rho: float, xi: float) -> np.ndarray:
        """
        Generate Heston model paths
        
        Args:
            S0: Initial stock price
            T: Time to expiry (years)
            sigma0: Initial volatility
            kappa: Mean reversion speed
            theta: Long-term volatility
            rho: Correlation
            xi: Volatility of volatility
        
        Returns:
            Stock price path matrix
        """
        dt = T / self.n_steps
        
        # Generate correlated random numbers
        Z1 = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        
        # Initialize paths
        S = np.zeros((self.n_simulations, self.n_steps + 1))
        sigma = np.zeros((self.n_simulations, self.n_steps + 1))
        
        S[:, 0] = S0
        sigma[:, 0] = sigma0
        
        # Generate paths
        for i in range(self.n_steps):
            # Volatility process
            sigma[:, i + 1] = sigma[:, i] + kappa * (theta - sigma[:, i]) * dt + xi * np.sqrt(sigma[:, i]) * np.sqrt(dt) * Z2[:, i]
            sigma[:, i + 1] = np.maximum(sigma[:, i + 1], 0)  # Ensure volatility is positive
            
            # Stock price process
            S[:, i + 1] = S[:, i] * np.exp((r - 0.5 * sigma[:, i]**2) * dt + sigma[:, i] * np.sqrt(dt) * Z1[:, i])
        
        return S
    
    def price_european_option(self, S0: float, K: float, T: float, r: float, sigma: float,
                            option_type: str = 'call', model: str = 'gbm') -> Dict:
        """
        Calculate European option price
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type
            model: Model type ('gbm' or 'heston')
        
        Returns:
            Dictionary containing price and statistical information
        """
        if model.lower() == 'gbm':
            paths = self._generate_gbm_paths(S0, T, r, sigma)
        else:
            # Use default parameters for Heston model
            paths = self._generate_heston_paths(S0, T, r, sigma, 2.0, sigma**2, -0.5, 0.3)
        
        # Calculate payoff at expiry
        final_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:  # put
            payoffs = np.maximum(K - final_prices, 0)
        
        # Calculate option price
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        # Calculate standard error
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        # Calculate 95% confidence interval
        confidence_interval = 1.96 * std_error
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': confidence_interval,
            'paths': paths,
            'payoffs': payoffs
        }
    
    def price_asian_option(self, S0: float, K: float, T: float, r: float, sigma: float,
                          option_type: str = 'call', averaging_type: str = 'arithmetic') -> Dict:
        """
        Calculate Asian option price
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type
            averaging_type: Averaging type ('arithmetic' or 'geometric')
        
        Returns:
            Dictionary containing price and statistical information
        """
        paths = self._generate_gbm_paths(S0, T, r, sigma)
        
        # Calculate average price
        if averaging_type.lower() == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        
        # Calculate payoff
        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:  # put
            payoffs = np.maximum(K - avg_prices, 0)
        
        # Calculate option price
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'paths': paths,
            'avg_prices': avg_prices,
            'payoffs': payoffs
        }
    
    def price_barrier_option(self, S0: float, K: float, T: float, r: float, sigma: float,
                           barrier: float, option_type: str = 'call', 
                           barrier_type: str = 'down_and_out') -> Dict:
        """
        Calculate barrier option price
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            barrier: Barrier price
            option_type: Option type
            barrier_type: Barrier type ('down_and_out', 'up_and_out', 'down_and_in', 'up_and_in')
        
        Returns:
            Dictionary containing price and statistical information
        """
        paths = self._generate_gbm_paths(S0, T, r, sigma)
        
        # Check barrier condition
        if barrier_type in ['down_and_out', 'down_and_in']:
            barrier_hit = np.any(paths <= barrier, axis=1)
        else:  # up_and_out, up_and_in
            barrier_hit = np.any(paths >= barrier, axis=1)
        
        # Calculate payoff at expiry
        final_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:  # put
            payoffs = np.maximum(K - final_prices, 0)
        
        # Adjust payoff based on barrier type
        if barrier_type in ['down_and_out', 'up_and_out']:
            payoffs = np.where(barrier_hit, 0, payoffs)
        else:  # down_and_in, up_and_in
            payoffs = np.where(barrier_hit, payoffs, 0)
        
        # Calculate option price
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'paths': paths,
            'barrier_hit': barrier_hit,
            'payoffs': payoffs
        }
    
    def price_lookback_option(self, S0: float, T: float, r: float, sigma: float,
                            option_type: str = 'call', lookback_type: str = 'floating') -> Dict:
        """
        Calculate lookback option price
        
        Args:
            S0: Initial stock price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type
            lookback_type: Lookback type ('floating' or 'fixed')
        
        Returns:
            Dictionary containing price and statistical information
        """
        paths = self._generate_gbm_paths(S0, T, r, sigma)
        
        if lookback_type.lower() == 'floating':
            # Floating strike lookback option
            if option_type.lower() == 'call':
                # Strike price is the minimum of the path
                strike_prices = np.min(paths, axis=1)
                payoffs = paths[:, -1] - strike_prices
            else:  # put
                # Strike price is the maximum of the path
                strike_prices = np.max(paths, axis=1)
                payoffs = strike_prices - paths[:, -1]
        else:
            # Fixed strike lookback option
            K = S0  # Use initial stock price as strike price
            if option_type.lower() == 'call':
                max_prices = np.max(paths, axis=1)
                payoffs = np.maximum(max_prices - K, 0)
            else:  # put
                min_prices = np.min(paths, axis=1)
                payoffs = np.maximum(K - min_prices, 0)
        
        # Calculate option price
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(self.n_simulations)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'paths': paths,
            'payoffs': payoffs
        }
    
    def calculate_greeks(self, S0: float, K: float, T: float, r: float, sigma: float,
                        option_type: str = 'call', dS: float = 0.01, 
                        dsigma: float = 0.01, dr: float = 0.01) -> Dict:
        """
        Calculate Greeks (using finite difference method)
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type
            dS: Stock price change
            dsigma: Volatility change
            dr: Interest rate change
        
        Returns:
            Dictionary containing Greeks
        """
        # Base price
        base_result = self.price_european_option(S0, K, T, r, sigma, option_type)
        base_price = base_result['price']
        
        # Delta
        price_up = self.price_european_option(S0 + dS, K, T, r, sigma, option_type)['price']
        price_down = self.price_european_option(S0 - dS, K, T, r, sigma, option_type)['price']
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma
        gamma = (price_up + price_down - 2 * base_price) / (dS ** 2)
        
        # Vega
        price_vol_up = self.price_european_option(S0, K, T, r, sigma + dsigma, option_type)['price']
        price_vol_down = self.price_european_option(S0, K, T, r, sigma - dsigma, option_type)['price']
        vega = (price_vol_up - price_vol_down) / (2 * dsigma)
        
        # Rho
        price_r_up = self.price_european_option(S0, K, T, r + dr, sigma, option_type)['price']
        price_r_down = self.price_european_option(S0, K, T, r - dr, sigma, option_type)['price']
        rho = (price_r_up - price_r_down) / (2 * dr)
        
        # Theta
        dt = 0.01
        price_time = self.price_european_option(S0, K, T - dt, r, sigma, option_type)['price']
        theta = (price_time - base_price) / dt
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    def variance_reduction_antithetic(self, S0: float, K: float, T: float, r: float, sigma: float,
                                   option_type: str = 'call') -> Dict:
        """
        Use antithetic variates for variance reduction
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Option type
        
        Returns:
            Dictionary containing price and statistical information
        """
        dt = T / self.n_steps
        
        # Generate random numbers
        Z = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        
        # Calculate paths
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        # Forward paths
        log_returns = drift + diffusion * Z
        cumulative_returns = np.cumsum(log_returns, axis=1)
        paths1 = S0 * np.exp(cumulative_returns)
        
        # Antithetic paths
        log_returns_antithetic = drift + diffusion * (-Z)
        cumulative_returns_antithetic = np.cumsum(log_returns_antithetic, axis=1)
        paths2 = S0 * np.exp(cumulative_returns_antithetic)
        
        # Calculate payoffs
        final_prices1 = paths1[:, -1]
        final_prices2 = paths2[:, -1]
        
        if option_type.lower() == 'call':
            payoffs1 = np.maximum(final_prices1 - K, 0)
            payoffs2 = np.maximum(final_prices2 - K, 0)
        else:  # put
            payoffs1 = np.maximum(K - final_prices1, 0)
            payoffs2 = np.maximum(K - final_prices2, 0)
        
        # Use antithetic variates
        payoffs_antithetic = (payoffs1 + payoffs2) / 2
        
        # Calculate option price
        option_price = np.exp(-r * T) * np.mean(payoffs_antithetic)
        std_error = np.exp(-r * T) * np.std(payoffs_antithetic) / np.sqrt(self.n_simulations)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'paths1': paths1,
            'paths2': paths2,
            'payoffs_antithetic': payoffs_antithetic
        }
