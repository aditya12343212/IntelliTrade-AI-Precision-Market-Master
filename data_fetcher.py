"""
Stock data fetcher module for live hourly data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import STOCK_SYMBOLS, DATA_INTERVAL, LOOKBACK_DAYS
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataFetcher:
    def __init__(self, symbols=STOCK_SYMBOLS):
        self.symbols = symbols
        self.interval = DATA_INTERVAL
        self.lookback_days = LOOKBACK_DAYS
    
    def fetch_live_data(self, symbol, period="30d"):
        """
        Fetch live hourly data for a given symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=self.interval)
            print(data)
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_symbols(self, symbols=None):
        """
        Fetch data for multiple symbols
        """
        if symbols is None:
            symbols = self.symbols
        
        data_dict = {}
        for symbol in symbols:
            data = self.fetch_live_data(symbol)
            if data is not None:
                data_dict[symbol] = data
        
        return data_dict
    
    def _add_technical_indicators(self, data):
        """
        Add technical indicators to the data
        """
        # Simple Moving Averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Price change indicators
        data['Price_Change'] = data['Close'].pct_change()
        data['High_Low_Ratio'] = (data['High'] - data['Low']) / data['Close']
        
        return data
    
    def _calculate_rsi(self, prices, window=14):
        """
        Calculate Relative Strength Index
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_latest_price(self, symbol):
        """
        Get the latest price for a symbol
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None
    
    def is_market_open(self):
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et <= market_close

if __name__ == "__main__":
    fetcher = StockDataFetcher()
    
    # Test with a single symbol
    data = fetcher.fetch_live_data("AAPL")
    if data is not None:
        print(f"AAPL data shape: {data.shape}")
        print(data.tail())
        
    # Test market status
    print(f"Market open: {fetcher.is_market_open()}")