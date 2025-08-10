"""
Options Chain Data Loader
Fetch options chain data from Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import config


class OptionChainLoader:
    """Options chain data loader class"""
    
    def __init__(self):
        pass
    
    def get_option_chain(self, symbol: str, expiration_date: str = None) -> Dict:
        """
        Get options chain data
        
        Args:
            symbol: Stock symbol
            expiration_date: Expiration date (YYYY-MM-DD), if None get all expirations
        
        Returns:
            Dictionary containing call and put options
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if expiration_date:
                # Get options chain for specific expiration date
                options = ticker.option_chain(expiration_date)
                return {
                    'calls': options.calls,
                    'puts': options.puts,
                    'expiration': expiration_date
                }
            else:
                # Get all available expiration dates
                expiration_dates = ticker.options
                if not expiration_dates:
                    return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
                
                # Get the nearest expiration date
                next_expiry = expiration_dates[0]
                options = ticker.option_chain(next_expiry)
                
                return {
                    'calls': options.calls,
                    'puts': options.puts,
                    'expiration': next_expiry,
                    'all_expirations': expiration_dates
                }
                
        except Exception as e:
            print(f"Error fetching options chain for {symbol}: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    
    def get_all_expirations(self, symbol: str) -> List[str]:
        """
        Get all available expiration dates
        
        Args:
            symbol: Stock symbol
        
        Returns:
            List of expiration dates
        """
        try:
            ticker = yf.Ticker(symbol)
            return ticker.options
        except Exception as e:
            print(f"Error fetching expiration dates for {symbol}: {e}")
            return []
    
    def filter_options(self, options_df: pd.DataFrame, 
                      min_volume: int = None, 
                      min_open_interest: int = None,
                      max_days_to_expiry: int = None) -> pd.DataFrame:
        """
        Filter options data
        
        Args:
            options_df: Options DataFrame
            min_volume: Minimum volume
            min_open_interest: Minimum open interest
            max_days_to_expiry: Maximum days to expiry
        
        Returns:
            Filtered DataFrame
        """
        if options_df.empty:
            return options_df
        
        filtered_df = options_df.copy()
        
        if min_volume is not None:
            filtered_df = filtered_df[filtered_df['volume'] >= min_volume]
        
        if min_open_interest is not None:
            filtered_df = filtered_df[filtered_df['openInterest'] >= min_open_interest]
        
        if max_days_to_expiry is not None:
            # Calculate days to expiry
            if 'lastTradeDate' in filtered_df.columns:
                filtered_df['days_to_expiry'] = (
                    pd.to_datetime(filtered_df['lastTradeDate']) - pd.Timestamp.now()
                ).dt.days
                filtered_df = filtered_df[filtered_df['days_to_expiry'] <= max_days_to_expiry]
        
        return filtered_df
    
    def calculate_implied_volatility(self, options_df: pd.DataFrame, 
                                   current_price: float, 
                                   risk_free_rate: float) -> pd.DataFrame:
        """
        Calculate implied volatility (simplified version)
        
        Args:
            options_df: Options DataFrame
            current_price: Current stock price
            risk_free_rate: Risk-free rate
        
        Returns:
            DataFrame with implied volatility
        """
        if options_df.empty:
            return options_df
        
        df = options_df.copy()
        
        # Simplified implied volatility calculation (should use more precise methods in practice)
        for idx, row in df.iterrows():
            try:
                # Use Black-Scholes to back out implied volatility
                # This uses simplified calculation, should use scipy.optimize in practice
                S = current_price
                K = row['strike']
                T = (pd.to_datetime(row['lastTradeDate']) - pd.Timestamp.now()).days / 365
                r = risk_free_rate
                option_price = row['lastPrice']
                
                # Simplified implied volatility estimation
                if T > 0:
                    # Use approximation formula
                    moneyness = np.log(S / K)
                    time_value = option_price - max(S - K, 0) if row['contractSymbol'].endswith('C') else max(K - S, 0)
                    
                    if time_value > 0:
                        implied_vol = np.sqrt(2 * np.pi / T) * time_value / S
                        df.at[idx, 'implied_volatility'] = min(max(implied_vol, 0.01), 2.0)
                    else:
                        df.at[idx, 'implied_volatility'] = 0.2
                else:
                    df.at[idx, 'implied_volatility'] = 0.2
                    
            except:
                df.at[idx, 'implied_volatility'] = 0.2
        
        return df
    
    def get_atm_options(self, symbol: str, expiration_date: str = None, 
                       tolerance: float = 0.05) -> Dict:
        """
        Get at-the-money options
        
        Args:
            symbol: Stock symbol
            expiration_date: Expiration date
            tolerance: ATM tolerance (percentage)
        
        Returns:
            ATM options dictionary
        """
        try:
            # Get current stock price
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get('regularMarketPrice', 100)
            
            # Get options chain
            option_chain = self.get_option_chain(symbol, expiration_date)
            
            if option_chain['calls'].empty or option_chain['puts'].empty:
                return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
            
            # Filter ATM options
            atm_calls = option_chain['calls'][
                abs(option_chain['calls']['strike'] - current_price) / current_price <= tolerance
            ]
            
            atm_puts = option_chain['puts'][
                abs(option_chain['puts']['strike'] - current_price) / current_price <= tolerance
            ]
            
            return {
                'calls': atm_calls,
                'puts': atm_puts,
                'current_price': current_price
            }
            
        except Exception as e:
            print(f"Error fetching ATM options: {e}")
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    
    def get_option_summary(self, symbol: str) -> Dict:
        """
        Get options summary information
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Options summary dictionary
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get all expiration dates
            expirations = self.get_all_expirations(symbol)
            
            # Get the nearest options chain
            if expirations:
                option_chain = self.get_option_chain(symbol, expirations[0])
                
                summary = {
                    'symbol': symbol,
                    'current_price': info.get('regularMarketPrice', 0),
                    'market_cap': info.get('marketCap', 0),
                    'volume': info.get('volume', 0),
                    'available_expirations': len(expirations),
                    'next_expiration': expirations[0] if expirations else None,
                    'total_call_options': len(option_chain['calls']),
                    'total_put_options': len(option_chain['puts']),
                    'avg_call_volume': option_chain['calls']['volume'].mean() if not option_chain['calls'].empty else 0,
                    'avg_put_volume': option_chain['puts']['volume'].mean() if not option_chain['puts'].empty else 0
                }
            else:
                summary = {
                    'symbol': symbol,
                    'current_price': info.get('regularMarketPrice', 0),
                    'market_cap': info.get('marketCap', 0),
                    'volume': info.get('volume', 0),
                    'available_expirations': 0,
                    'next_expiration': None,
                    'total_call_options': 0,
                    'total_put_options': 0,
                    'avg_call_volume': 0,
                    'avg_put_volume': 0
                }
            
            return summary
            
        except Exception as e:
            print(f"Error fetching options summary: {e}")
            return {}
