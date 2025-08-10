"""
Data Loader
Fetch stock and interest rate data from Yahoo Finance and FRED API
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os
import pickle
from fredapi import Fred
import config


class DataLoader:
    """Data loader class"""
    
    def __init__(self, cache_dir: str = config.DATA_CACHE_DIR):
        self.cache_dir = cache_dir
        self.fred = Fred(api_key=config.FRED_API_KEY) if config.FRED_API_KEY else None
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, filename: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, filename)
    
    def _load_from_cache(self, filename: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        cache_path = self._get_cache_path(filename)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def _save_to_cache(self, data: pd.DataFrame, filename: str):
        """Save data to cache"""
        cache_path = self._get_cache_path(filename)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    
    def get_stock_data(self, symbol: str, start_date: str = None, 
                      end_date: str = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Get stock historical data
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cache
        
        Returns:
            DataFrame containing OHLCV data
        """
        if start_date is None:
            start_date = config.DEFAULT_START_DATE
        if end_date is None:
            end_date = config.DEFAULT_END_DATE
        
        cache_filename = f"{symbol}_{start_date}_{end_date}.pkl"
        
        if use_cache:
            cached_data = self._load_from_cache(cache_filename)
            if cached_data is not None:
                return cached_data
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"Unable to fetch data for {symbol}")
            
            # Calculate daily returns
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # Calculate moving averages
            data['MA_20'] = data['Close'].rolling(window=20).mean()
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            
            # Calculate volatility
            data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            if use_cache:
                self._save_to_cache(data, cache_filename)
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_risk_free_rate(self, start_date: str = None, end_date: str = None) -> float:
        """
        Get risk-free rate (US 10-year Treasury yield)
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            Risk-free rate
        """
        if self.fred is None:
            print("Note: FRED API key not set, using default risk-free rate")
            return config.DEFAULT_RISK_FREE_RATE
        
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get 10-year Treasury yield
            rate_data = self.fred.get_series('GS10', start_date, end_date)
            
            if not rate_data.empty:
                return rate_data.iloc[-1] / 100  # Convert to decimal
            else:
                return config.DEFAULT_RISK_FREE_RATE
                
        except Exception as e:
            print(f"Error fetching risk-free rate: {e}")
            return config.DEFAULT_RISK_FREE_RATE
    
    def get_vix_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get VIX index data
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            VIX data DataFrame
        """
        if start_date is None:
            start_date = config.DEFAULT_START_DATE
        if end_date is None:
            end_date = config.DEFAULT_END_DATE
        
        cache_filename = f"VIX_{start_date}_{end_date}.pkl"
        
        cached_data = self._load_from_cache(cache_filename)
        if cached_data is not None:
            return cached_data
        
        try:
            vix = yf.Ticker('^VIX')
            data = vix.history(start=start_date, end=end_date)
            
            if not data.empty and use_cache:
                self._save_to_cache(data, cache_filename)
            
            return data
            
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return pd.DataFrame()
    
    def calculate_historical_volatility(self, symbol: str, window: int = 252) -> float:
        """
        Calculate historical volatility
        
        Args:
            symbol: Stock symbol
            window: Calculation window (days)
        
        Returns:
            Annualized volatility
        """
        data = self.get_stock_data(symbol)
        if data.empty:
            return config.DEFAULT_VOLATILITY
        
        returns = data['Returns'].dropna()
        if len(returns) < window:
            window = len(returns)
        
        return returns.tail(window).std() * np.sqrt(252)
    
    def get_multiple_stocks(self, symbols: List[str], start_date: str = None, 
                          end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary of stock data
        """
        data_dict = {}
        for symbol in symbols:
            data_dict[symbol] = self.get_stock_data(symbol, start_date, end_date)
        
        return data_dict
