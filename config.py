"""
Configuration file for the ML Stock Trading System
"""

# Stock symbols to monitor
STOCK_SYMBOLS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
    'NVDA', 'META', 'NFLX', 'AMD', 'CRM',
    'UBER', 'PYPL', 'ADBE', 'INTC', 'ORCL',
    'IBM', 'SHOP', 'SQ', 'ZOOM', 'DOCU'
]

# Trading parameters
SUPPORT_RESISTANCE_WINDOW = 20  # Number of periods to look back
MIN_TOUCHES = 2  # Minimum touches to confirm S/R level
BREAKOUT_THRESHOLD = 0.02  # 2% breakout threshold
VOLUME_MULTIPLIER = 1.5  # Volume confirmation multiplier

# Risk management
STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.10  # 10% take profit
MAX_POSITION_SIZE = 0.1  # Maximum 10% of portfolio per position

# Data settings
DATA_INTERVAL = '1h'  # Hourly data
LOOKBACK_DAYS = 30  # Days of historical data to fetch

# Notification settings
TELEGRAM_BOT_TOKEN = ""  # Add your Telegram bot token
TELEGRAM_CHAT_ID = ""    # Add your Telegram chat ID

# Model settings
MODEL_RETRAIN_FREQUENCY = 24  # Retrain model every 24 hours
PREDICTION_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for signals