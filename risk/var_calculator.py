"""
VaR (Value at Risk) Calculator
Implements multiple VaR calculation methods
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Optional, Tuple
import config


class VaRCalculator:
    """VaR Calculator Class"""
    
    def __init__(self, confidence_level: float = config.VAR_CONFIDENCE_LEVEL):
        self.confidence_level = confidence_level
    
    def historical_var(self, returns: pd.Series, confidence_level: float = None, 
                      time_horizon: int = 1) -> Dict:
        """
        Historical Simulation VaR
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            time_horizon: Time horizon (days)
        
        Returns:
            VaR result dictionary
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Clean data
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {'var': 0, 'cvar': 0, 'confidence_level': confidence_level}
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(returns_clean, var_percentile)
        
        # Calculate CVaR (Conditional VaR)
        cvar = returns_clean[returns_clean <= var].mean()
        
        # Annualized VaR
        var_annualized = var * np.sqrt(252 * time_horizon)
        cvar_annualized = cvar * np.sqrt(252 * time_horizon)
        
        return {
            'var': var,
            'cvar': cvar,
            'var_annualized': var_annualized,
            'cvar_annualized': cvar_annualized,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'data_points': len(returns_clean)
        }
    
    def parametric_var(self, returns: pd.Series, confidence_level: float = None,
                      time_horizon: int = 1) -> Dict:
        """
        Parametric VaR (assuming normal distribution)
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            time_horizon: Time horizon (days)
        
        Returns:
            VaR result dictionary
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Clean data
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {'var': 0, 'cvar': 0, 'confidence_level': confidence_level}
        
        # Calculate statistics
        mean_return = returns_clean.mean()
        std_return = returns_clean.std()
        
        # Calculate quantile
        z_score = norm.ppf(1 - confidence_level)
        
        # Calculate VaR
        var = mean_return - z_score * std_return
        
        # Calculate CVaR
        cvar = mean_return - (norm.pdf(z_score) / (1 - confidence_level)) * std_return
        
        # Annualized VaR
        var_annualized = var * np.sqrt(252 * time_horizon)
        cvar_annualized = cvar * np.sqrt(252 * time_horizon)
        
        return {
            'var': var,
            'cvar': cvar,
            'var_annualized': var_annualized,
            'cvar_annualized': cvar_annualized,
            'mean_return': mean_return,
            'std_return': std_return,
            'z_score': z_score,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'data_points': len(returns_clean)
        }
    
    def monte_carlo_var(self, returns: pd.Series, confidence_level: float = None,
                       time_horizon: int = 1, n_simulations: int = 10000) -> Dict:
        """
        Monte Carlo VaR
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            time_horizon: Time horizon (days)
            n_simulations: Number of simulations
        
        Returns:
            VaR result dictionary
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Clean data
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return {'var': 0, 'cvar': 0, 'confidence_level': confidence_level}
        
        # Estimate parameters
        mean_return = returns_clean.mean()
        std_return = returns_clean.std()
        
        # Generate simulated returns
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(simulated_returns, var_percentile)
        
        # Calculate CVaR
        cvar = simulated_returns[simulated_returns <= var].mean()
        
        # Annualized VaR
        var_annualized = var * np.sqrt(252 * time_horizon)
        cvar_annualized = cvar * np.sqrt(252 * time_horizon)
        
        return {
            'var': var,
            'cvar': cvar,
            'var_annualized': var_annualized,
            'cvar_annualized': cvar_annualized,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'n_simulations': n_simulations,
            'simulated_returns': simulated_returns
        }
    
    def portfolio_var(self, returns_matrix: pd.DataFrame, weights: List[float],
                     confidence_level: float = None, time_horizon: int = 1,
                     method: str = 'parametric') -> Dict:
        """
        Portfolio VaR
        
        Args:
            returns_matrix: Returns matrix (each column is an asset)
            weights: Weight list
            confidence_level: Confidence level
            time_horizon: Time horizon (days)
            method: Calculation method ('parametric', 'historical', 'monte_carlo')
        
        Returns:
            Portfolio VaR result dictionary
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Ensure weights sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Calculate portfolio returns
        portfolio_returns = (returns_matrix * weights).sum(axis=1)
        
        # Calculate VaR based on method
        if method.lower() == 'parametric':
            return self.parametric_var(portfolio_returns, confidence_level, time_horizon)
        elif method.lower() == 'historical':
            return self.historical_var(portfolio_returns, confidence_level, time_horizon)
        elif method.lower() == 'monte_carlo':
            return self.monte_carlo_var(portfolio_returns, confidence_level, time_horizon)
        else:
            raise ValueError("method must be 'parametric', 'historical', or 'monte_carlo'")
    
    def var_decomposition(self, returns_matrix: pd.DataFrame, weights: List[float],
                         confidence_level: float = None, time_horizon: int = 1) -> Dict:
        """
        VaR Decomposition (Marginal VaR and Component VaR)
        
        Args:
            returns_matrix: Returns matrix
            weights: Weight list
            confidence_level: Confidence level
            time_horizon: Time horizon (days)
        
        Returns:
            VaR decomposition result dictionary
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Calculate portfolio VaR
        portfolio_var_result = self.portfolio_var(returns_matrix, weights, confidence_level, time_horizon)
        portfolio_var = portfolio_var_result['var_annualized']
        
        # Calculate covariance matrix
        cov_matrix = returns_matrix.cov() * 252  # Annualized
        
        # Convert weights to numpy array
        weights_array = np.array(weights)
        
        # Calculate portfolio standard deviation
        portfolio_std = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
        
        # Calculate marginal VaR
        marginal_var = (cov_matrix @ weights_array) / portfolio_std * norm.ppf(confidence_level)
        
        # Calculate component VaR
        component_var = weights_array * marginal_var
        
        # Calculate contribution
        contribution = component_var / portfolio_var
        
        return {
            'portfolio_var': portfolio_var,
            'marginal_var': marginal_var,
            'component_var': component_var,
            'contribution': contribution,
            'weights': weights,
            'confidence_level': confidence_level
        }
    
    def stress_testing(self, returns: pd.Series, stress_scenarios: Dict[str, float],
                      confidence_level: float = None) -> Dict:
        """
        Stress Testing
        
        Args:
            returns: Return series
            stress_scenarios: Stress scenario dictionary
            confidence_level: Confidence level
        
        Returns:
            Stress testing result dictionary
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        results = {}
        
        # Base VaR
        base_var = self.parametric_var(returns, confidence_level)
        results['base_var'] = base_var['var_annualized']
        
        # Stress scenario VaR
        for scenario_name, stress_factor in stress_scenarios.items():
            # Adjust returns (e.g., increase volatility)
            stressed_returns = returns * stress_factor
            stressed_var = self.parametric_var(stressed_returns, confidence_level)
            results[f'{scenario_name}_var'] = stressed_var['var_annualized']
            results[f'{scenario_name}_impact'] = stressed_var['var_annualized'] - base_var['var_annualized']
        
        return results
    
    def backtest_var(self, returns: pd.Series, var_estimates: pd.Series,
                    confidence_level: float = None) -> Dict:
        """
        VaR Backtesting
        
        Args:
            returns: Actual returns
            var_estimates: VaR estimates
            confidence_level: Confidence level
        
        Returns:
            Backtesting result dictionary
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Calculate violation count
        violations = returns < var_estimates
        n_violations = violations.sum()
        n_observations = len(returns)
        violation_rate = n_violations / n_observations
        
        # Theoretical violation rate
        expected_violation_rate = 1 - confidence_level
        
        # Kupiec test statistic
        if violation_rate > 0 and violation_rate < 1:
            kupiec_stat = -2 * np.log(((1 - expected_violation_rate) ** (n_observations - n_violations) * 
                                      expected_violation_rate ** n_violations) /
                                     ((1 - violation_rate) ** (n_observations - n_violations) * 
                                      violation_rate ** n_violations))
        else:
            kupiec_stat = np.inf
        
        # Maximum consecutive violations
        max_consecutive_violations = 0
        current_consecutive = 0
        for violation in violations:
            if violation:
                current_consecutive += 1
                max_consecutive_violations = max(max_consecutive_violations, current_consecutive)
            else:
                current_consecutive = 0
        
        return {
            'n_violations': n_violations,
            'n_observations': n_observations,
            'violation_rate': violation_rate,
            'expected_violation_rate': expected_violation_rate,
            'kupiec_statistic': kupiec_stat,
            'max_consecutive_violations': max_consecutive_violations,
            'is_reliable': abs(violation_rate - expected_violation_rate) < 0.01
        }
    
    def rolling_var(self, returns: pd.Series, window: int = 252,
                   confidence_level: float = None, method: str = 'parametric') -> pd.Series:
        """
        Rolling VaR
        
        Args:
            returns: Return series
            window: Rolling window size
            confidence_level: Confidence level
            method: Calculation method
        
        Returns:
            Rolling VaR series
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        rolling_var = pd.Series(index=returns.index, dtype=float)
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            
            if method.lower() == 'parametric':
                var_result = self.parametric_var(window_returns, confidence_level)
            elif method.lower() == 'historical':
                var_result = self.historical_var(window_returns, confidence_level)
            else:
                raise ValueError("method must be 'parametric' or 'historical'")
            
            rolling_var.iloc[i] = var_result['var_annualized']
        
        return rolling_var
