"""
Model Test File
Tests the accuracy of various pricing models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from models.black_scholes import BlackScholes
from models.binomial_tree import BinomialTree
from models.monte_carlo import MonteCarloPricing

class TestPricingModels(unittest.TestCase):
    """Pricing Models Test Class"""
    
    def setUp(self):
        """Set up test environment"""
        self.bs = BlackScholes()
        self.binomial = BinomialTree(steps=100)
        self.mc = MonteCarloPricing(n_simulations=10000)
        
        # Test parameters
        self.S = 100
        self.K = 100
        self.T = 0.25
        self.r = 0.05
        self.sigma = 0.2
    
    def test_black_scholes_call(self):
        """Test Black-Scholes call option"""
        price = self.bs.price_call(self.S, self.K, self.T, self.r, self.sigma)
        
        # Check if price is positive
        self.assertGreater(price, 0)
        
        # Check if price is reasonable (should be greater than intrinsic value)
        intrinsic_value = max(self.S - self.K, 0)
        self.assertGreaterEqual(price, intrinsic_value)
    
    def test_black_scholes_put(self):
        """Test Black-Scholes put option"""
        price = self.bs.price_put(self.S, self.K, self.T, self.r, self.sigma)
        
        # Check if price is positive
        self.assertGreater(price, 0)
        
        # Check if price is reasonable (should be greater than intrinsic value)
        intrinsic_value = max(self.K - self.S, 0)
        self.assertGreaterEqual(price, intrinsic_value)
    
    def test_put_call_parity(self):
        """Test put-call parity relationship"""
        call_price = self.bs.price_call(self.S, self.K, self.T, self.r, self.sigma)
        put_price = self.bs.price_put(self.S, self.K, self.T, self.r, self.sigma)
        
        # Put-call parity: C - P = S - K*exp(-r*T)
        left_side = call_price - put_price
        right_side = self.S - self.K * np.exp(-self.r * self.T)
        
        # Allow small numerical errors
        self.assertAlmostEqual(left_side, right_side, places=6)
    
    def test_binomial_tree(self):
        """Test binomial tree model"""
        result = self.binomial.price_option(self.S, self.K, self.T, self.r, self.sigma, 'call')
        
        # Check result structure
        self.assertIn('price', result)
        self.assertIn('price_tree', result)
        self.assertIn('option_tree', result)
        
        # Check if price is positive
        self.assertGreater(result['price'], 0)
    
    def test_monte_carlo(self):
        """Test Monte Carlo model"""
        result = self.mc.price_european_option(self.S, self.K, self.T, self.r, self.sigma, 'call')
        
        # Check result structure
        self.assertIn('price', result)
        self.assertIn('std_error', result)
        self.assertIn('paths', result)
        
        # Check if price is positive
        self.assertGreater(result['price'], 0)
        
        # Check if standard error is positive
        self.assertGreater(result['std_error'], 0)
    
    def test_model_convergence(self):
        """Test model convergence"""
        # Black-Scholes as benchmark
        bs_price = self.bs.price_call(self.S, self.K, self.T, self.r, self.sigma)
        
        # Binomial tree should be close to Black-Scholes
        binomial_price = self.binomial.price_option(self.S, self.K, self.T, self.r, self.sigma, 'call')['price']
        self.assertAlmostEqual(bs_price, binomial_price, places=2)
        
        # Monte Carlo should be within reasonable range
        mc_price = self.mc.price_european_option(self.S, self.K, self.T, self.r, self.sigma, 'call')['price']
        # Allow 5% error
        self.assertAlmostEqual(bs_price, mc_price, delta=bs_price * 0.05)
    
    def test_greeks(self):
        """Test Greeks calculation"""
        greeks = self.bs.calculate_all_greeks('call', self.S, self.K, self.T, self.r, self.sigma)
        
        # Check if all Greeks exist
        required_greeks = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho']
        for greek in required_greeks:
            self.assertIn(greek, greeks)
        
        # Check Delta range
        self.assertGreaterEqual(greeks['delta'], 0)
        self.assertLessEqual(greeks['delta'], 1)
        
        # Check Gamma is positive
        self.assertGreaterEqual(greeks['gamma'], 0)
    
    def test_american_options(self):
        """Test American options"""
        # American option price should be greater than or equal to European option
        european = self.binomial.price_option(self.S, self.K, self.T, self.r, self.sigma, 'call', american=False)
        american = self.binomial.price_option(self.S, self.K, self.T, self.r, self.sigma, 'call', american=True)
        
        self.assertGreaterEqual(american['price'], european['price'])
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Time to expiry is 0
        price = self.bs.price_call(self.S, self.K, 0, self.r, self.sigma)
        expected = max(self.S - self.K, 0)
        self.assertEqual(price, expected)
        
        # Volatility is 0
        price = self.bs.price_call(self.S, self.K, self.T, self.r, 0)
        # When volatility is 0, option price equals intrinsic value
        expected = max(self.S - self.K * np.exp(-self.r * self.T), 0)
        self.assertAlmostEqual(price, expected, places=6)
    
    def test_complex_options(self):
        """Test complex options"""
        # Asian options
        asian_result = self.mc.price_asian_option(self.S, self.K, self.T, self.r, self.sigma, 'call')
        self.assertGreater(asian_result['price'], 0)
        
        # Barrier options
        barrier = 90
        barrier_result = self.mc.price_barrier_option(self.S, self.K, self.T, self.r, self.sigma, barrier, 'call')
        self.assertGreaterEqual(barrier_result['price'], 0)
        
        # Lookback options
        lookback_result = self.mc.price_lookback_option(self.S, self.T, self.r, self.sigma, 'call')
        self.assertGreater(lookback_result['price'], 0)

if __name__ == '__main__':
    unittest.main()
