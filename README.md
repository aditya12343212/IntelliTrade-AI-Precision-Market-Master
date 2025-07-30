# ML Stock Trading System

A comprehensive machine learning-based stock trading system that analyzes live hourly stock data, identifies support and resistance levels, and generates automated trading signals with notifications.

## Features

### Core Components
- **Live Data Fetching**: Real-time hourly stock data using yfinance
- **Support/Resistance Detection**: Advanced algorithm using multiple methods
- **Machine Learning Model**: Pattern recognition and prediction system
- **Trading Strategy**: Comprehensive signal generation with risk management
- **Notification System**: Telegram alerts for trading signals
- **Web Dashboard**: Interactive visualization and monitoring

### Key Capabilities
- ✅ Monitors 20+ popular stocks (AAPL, GOOGL, MSFT, etc.)
- ✅ Identifies support and resistance levels automatically
- ✅ Generates buy/sell signals with confidence scores
- ✅ Risk management with stop-loss and take-profit levels
- ✅ Real-time alerts via Telegram
- ✅ Interactive web dashboard for analysis
- ✅ Automated model training and retraining

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd ml-stock-trading-system

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.py` to customize:

```python
# Add your Telegram bot credentials (optional)
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"

# Modify stock symbols to monitor
STOCK_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

# Adjust trading parameters
STOP_LOSS_PERCENTAGE = 0.05  # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.10  # 10% take profit
```

### 3. Run the System

#### Option A: Main Trading System
```bash
python main.py
```

#### Option B: Web Dashboard
```bash
python visualization.py
```
Then open: http://localhost:8050

#### Option C: Test Individual Components
```bash
# Test data fetching
python data_fetcher.py

# Test support/resistance detection
python support_resistance.py

# Test ML model
python ml_model.py

# Test trading strategy
python trading_strategy.py
```

## System Architecture

```
├── config.py              # Configuration settings
├── data_fetcher.py         # Live stock data fetching
├── support_resistance.py   # Support/resistance level detection
├── ml_model.py            # Machine learning model
├── trading_strategy.py     # Trading signal generation
├── notification_system.py # Alert notifications
├── visualization.py       # Web dashboard
├── main.py               # Main application
└── requirements.txt      # Dependencies
```

## How It Works

### 1. Data Collection
- Fetches live hourly data from Yahoo Finance
- Adds technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Monitors during market hours (9:30 AM - 4:00 PM EST)

### 2. Support/Resistance Detection
Uses multiple methods:
- **Local Extrema**: Mathematical peaks and valleys
- **Volume Profile**: High-volume price levels
- **Psychological Levels**: Round number levels ($50, $100, etc.)

### 3. Machine Learning
- **Features**: 25+ technical and fundamental indicators
- **Models**: Random Forest, Gradient Boosting, Logistic Regression
- **Labels**: Buy/Sell/Hold based on future price movements
- **Training**: Automatically retrains every 24 hours

### 4. Trading Strategy
Combines multiple signal sources:
- Technical analysis (MACD, RSI, Moving Averages)
- Support/resistance breakouts and bounces
- Volume confirmation
- ML model predictions

### 5. Risk Management
- Stop-loss orders (default: 5%)
- Take-profit targets (default: 10%)
- Position sizing limits (default: 10% max per position)
- Risk/reward ratio calculation

## Signal Types

### Buy Signals
- Support level bounce with volume
- Resistance breakout with confirmation
- MACD bullish crossover
- RSI oversold recovery
- ML model buy prediction (>70% confidence)

### Sell Signals
- Resistance rejection with volume
- Support breakdown with confirmation
- MACD bearish crossover
- RSI overbought reversal
- ML model sell prediction (>70% confidence)

## Notification System

### Telegram Setup
1. Create a Telegram bot via @BotFather
2. Get your bot token and chat ID
3. Add credentials to `config.py`

### Alert Types
- **Trading Signals**: Buy/sell recommendations with reasoning
- **Breakout Alerts**: Support/resistance level breaks
- **ML Predictions**: Model-based forecasts
- **Daily Summary**: Performance and system stats

## Web Dashboard

Access the interactive dashboard at `http://localhost:8050`

### Features
- Real-time price charts with support/resistance levels
- Technical indicators (RSI, MACD, Bollinger Bands)
- Recent trading signals table
- Performance metrics
- System status monitoring

## Configuration Options

### Stock Selection
```python
STOCK_SYMBOLS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA',
    'NVDA', 'META', 'NFLX', 'AMD', 'CRM'
    # Add more symbols as needed
]
```

### Trading Parameters
```python
SUPPORT_RESISTANCE_WINDOW = 20  # Lookback period
MIN_TOUCHES = 2                 # Minimum level confirmations
BREAKOUT_THRESHOLD = 0.02       # 2% breakout threshold
STOP_LOSS_PERCENTAGE = 0.05     # 5% stop loss
TAKE_PROFIT_PERCENTAGE = 0.10   # 10% take profit
```

### Model Settings
```python
MODEL_RETRAIN_FREQUENCY = 24         # Hours between retraining
PREDICTION_CONFIDENCE_THRESHOLD = 0.7 # Minimum confidence for signals
```

## Performance Tracking

The system automatically tracks:
- Total signals generated
- Win/loss ratios
- Average returns per trade
- Maximum drawdown
- Sharpe ratio
- System uptime and reliability

## Scheduling

Automated tasks:
- **Hourly**: Market analysis and signal generation
- **Daily 6 AM**: Model retraining
- **Daily 4:30 PM**: Performance summary

## Safety Features

- Market hours detection (no trading when market is closed)
- Rate limiting to avoid API restrictions
- Error handling and logging
- Graceful shutdown on interruption
- State persistence across restarts

## Disclaimer

⚠️ **Important**: This system is for educational and research purposes only. It is not financial advice. Always:

- Do your own research before making trades
- Never invest more than you can afford to lose
- Test thoroughly with paper trading first
- Understand the risks involved in algorithmic trading
- Consider consulting with a financial advisor

## Troubleshooting

### Common Issues

1. **No data fetched**: Check internet connection and symbol validity
2. **Model not training**: Ensure sufficient historical data (60+ days)
3. **Telegram not working**: Verify bot token and chat ID
4. **Dashboard not loading**: Check if port 8050 is available

### Logging

Check `trading_system.log` for detailed system logs and error messages.

## Contributing

Feel free to enhance the system by:
- Adding new technical indicators
- Improving the ML model
- Adding new data sources
- Creating additional visualization features
- Optimizing trading strategies

## License

This project is open source. Use at your own risk and responsibility.

---

**Happy Trading! 📈**