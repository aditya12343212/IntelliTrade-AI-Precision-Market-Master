"""
Main Application for ML Stock Trading System
"""

import time
import schedule # type: ignore
import threading
import logging
import json
import os
from datetime import datetime, timedelta
import pandas as pd

from data_fetcher import StockDataFetcher
from support_resistance import SupportResistanceDetector
from ml_model import StockMLModel
from trading_strategy import TradingStrategy
from notification_system import NotificationSystem
from config import (STOCK_SYMBOLS, MODEL_RETRAIN_FREQUENCY, 
                   PREDICTION_CONFIDENCE_THRESHOLD)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLTradingSystem:
    def __init__(self):
        self.data_fetcher = StockDataFetcher()
        self.sr_detector = SupportResistanceDetector()
        self.ml_model = StockMLModel()
        self.trading_strategy = TradingStrategy()
        self.notifier = NotificationSystem()
        
        self.symbols = STOCK_SYMBOLS
        self.model_path = "stock_ml_model.pkl"
        self.system_data_path = "system_data.json"
        
        self.is_running = False
        self.last_model_training = None
        self.system_stats = {
            'signals_generated': 0,
            'notifications_sent': 0,
            'system_uptime': 0,
            'last_update': None
        }
        
        # Load existing model if available
        self.load_system_state()
    
    def initialize_system(self):
        """
        Initialize the trading system
        """
        logger.info("Initializing ML Trading System...")
        
        # Test notification system
        self.notifier.test_notification_system()
        
        # Load or train ML model
        if not os.path.exists(self.model_path) or self.should_retrain_model():
            logger.info("Training ML model...")
            self.train_model()
        else:
            logger.info("Loading existing ML model...")
            self.ml_model.load_model(self.model_path)
        
        # Send startup notification
        startup_message = f"""
🚀 <b>TRADING SYSTEM STARTED</b> 🚀

<b>Status:</b> ✅ Online
<b>Monitoring:</b> {len(self.symbols)} symbols
<b>Model Status:</b> {'✅ Loaded' if self.ml_model.is_trained else '❌ Not Trained'}
<b>Start Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>Symbols:</b> {', '.join(self.symbols[:10])}{'...' if len(self.symbols) > 10 else ''}

🔄 System will check for signals every hour during market hours.
"""
        
        self.notifier.send_telegram_alert(startup_message.strip())
        logger.info("System initialization completed")
    
    def train_model(self):
        """
        Train or retrain the ML model
        """
        try:
            logger.info("Starting model training process...")
            
            # Collect training data from multiple symbols
            all_features = []
            all_labels = []
            
            for symbol in self.symbols[:5]:  # Train on first 5 symbols to save time
                logger.info(f"Fetching training data for {symbol}...")
                data = self.data_fetcher.fetch_live_data(symbol, period="90d")
                
                if data is not None and len(data) > 50:
                    # Get support/resistance levels
                    levels = self.sr_detector.find_support_resistance_levels(data)
                    
                    # Prepare features
                    features = self.ml_model.prepare_features(
                        data, levels['support_levels'], levels['resistance_levels']
                    )
                    
                    # Create labels
                    labels = self.ml_model.create_labels(features)
                    
                    if len(features) == len(labels):
                        all_features.append(features)
                        all_labels.extend(labels)
            
            if all_features and all_labels:
                # Combine all features
                combined_features = pd.concat(all_features, ignore_index=True)
                
                # Train the model
                success = self.ml_model.train_model(combined_features, all_labels)
                
                if success:
                    # Save the model
                    self.ml_model.save_model(self.model_path)
                    self.last_model_training = datetime.now()
                    
                    logger.info("Model training completed successfully")
                    
                    # Send training completion notification
                    training_message = f"""
🧠 <b>MODEL TRAINING COMPLETED</b> 🧠

<b>Training Data:</b> {len(combined_features)} samples
<b>Symbols Used:</b> {', '.join(self.symbols[:5])}
<b>Training Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ Model saved and ready for predictions.
"""
                    self.notifier.send_telegram_alert(training_message.strip())
                    
                    return True
                else:
                    logger.error("Model training failed")
                    return False
            else:
                logger.error("Insufficient training data")
                return False
                
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            return False
    
    def should_retrain_model(self):
        """
        Check if model should be retrained
        """
        if self.last_model_training is None:
            return True
        
        hours_since_training = (datetime.now() - self.last_model_training).total_seconds() / 3600
        return hours_since_training >= MODEL_RETRAIN_FREQUENCY
    
    def analyze_symbol(self, symbol):
        """
        Analyze a single symbol and generate signals
        """
        try:
            logger.info(f"Analyzing {symbol}...")
            
            # Fetch latest data
            data = self.data_fetcher.fetch_live_data(symbol)
            if data is None or len(data) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            
            # Find support/resistance levels
            levels = self.sr_detector.find_support_resistance_levels(data)
            
            # Check for breakouts
            breakouts = self.sr_detector.detect_breakouts(
                data, levels['support_levels'], levels['resistance_levels']
            )
            
            # Send breakout alerts
            for breakout in breakouts:
                self.notifier.send_breakout_alert(symbol, breakout)
            
            # ML predictions
            ml_predictions = None
            if self.ml_model.is_trained:
                features = self.ml_model.prepare_features(
                    data, levels['support_levels'], levels['resistance_levels']
                )
                ml_predictions = self.ml_model.predict(features.tail(1))
                
                # Send ML prediction alerts
                if ml_predictions:
                    for prediction in ml_predictions:
                        if prediction['confidence'] > PREDICTION_CONFIDENCE_THRESHOLD:
                            self.notifier.send_ml_prediction_alert(symbol, prediction)
            
            # Generate trading signals
            signal = self.trading_strategy.generate_signals(
                symbol, data, levels['support_levels'], 
                levels['resistance_levels'], ml_predictions
            )
            
            if signal and signal['action'] != 'hold' and signal['confidence'] > 0.7:
                # Send trading signal alert
                self.notifier.send_trading_signal_alert(signal)
                self.system_stats['signals_generated'] += 1
                self.system_stats['notifications_sent'] += 1
                
                logger.info(f"Signal generated for {symbol}: {signal['action']} with {signal['confidence']:.2f} confidence")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def run_analysis_cycle(self):
        """
        Run analysis cycle for all symbols
        """
        if not self.data_fetcher.is_market_open():
            logger.info("Market is closed, skipping analysis")
            return
        
        logger.info("Starting analysis cycle...")
        start_time = datetime.now()
        
        signals_generated = 0
        
        for symbol in self.symbols:
            try:
                signal = self.analyze_symbol(symbol)
                if signal and signal['action'] != 'hold':
                    signals_generated += 1
                
                # Small delay between symbols to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        # Update system stats
        cycle_duration = (datetime.now() - start_time).total_seconds()
        self.system_stats['last_update'] = datetime.now().isoformat()
        
        logger.info(f"Analysis cycle completed in {cycle_duration:.1f}s. Generated {signals_generated} signals.")
        
        # Save system state
        self.save_system_state()
    
    def send_daily_summary(self):
        """
        Send daily trading summary
        """
        try:
            # Get performance summary
            performance = self.trading_strategy.get_performance_summary()
            
            # Get notification stats
            notification_stats = self.notifier.get_notification_stats()
            
            summary_message = f"""
            📊 <b>DAILY TRADING SUMMARY</b> 📊

            <b>System Stats:</b>
            • Signals Generated: {self.system_stats['signals_generated']}
            • Notifications Sent: {self.system_stats['notifications_sent']}
            • Symbols Monitored: {len(self.symbols)}

            <b>Notifications:</b>
            • Total Sent: {notification_stats['total_notifications'] if notification_stats else 0}
            • Success Rate: {notification_stats['success_rate']:.1% if (notification_stats and notification_stats.get('total_notifications', 0) > 0) else 'N/A'}

            <b>Date:</b> {datetime.now():%Y-%m-%d}

            🔄 System continues monitoring for tomorrow's trading session.
            """

            
            self.notifier.send_telegram_alert(summary_message.strip())
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {str(e)}")
    
    def save_system_state(self):
        """
        Save system state to file
        """
        try:
            state_data = {
                'system_stats': self.system_stats,
                'last_model_training': self.last_model_training.isoformat() if self.last_model_training else None,
                'symbols': self.symbols
            }
            
            with open(self.system_data_path, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")
    
    def load_system_state(self):
        """
        Load system state from file
        """
        try:
            if os.path.exists(self.system_data_path):
                with open(self.system_data_path, 'r') as f:
                    state_data = json.load(f)
                
                self.system_stats = state_data.get('system_stats', self.system_stats)
                
                if state_data.get('last_model_training'):
                    self.last_model_training = datetime.fromisoformat(state_data['last_model_training'])
                
                logger.info("System state loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading system state: {str(e)}")
    
    def start_scheduler(self):
        """
        Start the scheduled tasks
        """
        # Schedule analysis every hour during market hours
        schedule.every().hour.at(":00").do(self.run_analysis_cycle)
        schedule.every().hour.at(":30").do(self.run_analysis_cycle)
        
        # Schedule model retraining daily at 6 AM
        schedule.every().day.at("06:00").do(self.train_model)
        
        # Schedule daily summary at 4:30 PM (after market close)
        schedule.every().day.at("16:30").do(self.send_daily_summary)
        
        # Run scheduler in a separate thread
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Scheduler started")
    
    def start(self):
        """
        Start the trading system
        """
        self.is_running = True
        
        try:
            # Initialize system
            self.initialize_system()
            
            # Start scheduler
            self.start_scheduler()
            
            # Run initial analysis
            self.run_analysis_cycle()
            
            logger.info("Trading system is now running...")
            
            # Keep the main thread alive
            while self.is_running:
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.stop()
        except Exception as e:
            logger.error(f"Critical error in main loop: {str(e)}")
            self.stop()
    
    def stop(self):
        """
        Stop the trading system
        """
        logger.info("Stopping trading system...")
        self.is_running = False
        
        # Send shutdown notification
        shutdown_message = f"""
🛑 <b>TRADING SYSTEM STOPPED</b> 🛑

<b>Status:</b> ❌ Offline
<b>Stop Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
<b>Total Signals:</b> {self.system_stats['signals_generated']}

System has been shut down gracefully.
"""
        
        self.notifier.send_telegram_alert(shutdown_message.strip())
        
        # Save final state
        self.save_system_state()
        self.notifier.save_notification_history()
        
        logger.info("Trading system stopped")

def main():
    """
    Main entry point
    """
    print("=" * 60)
    print("ML STOCK TRADING SYSTEM")
    print("=" * 60)
    print("Initializing system...")
    
    # Create and start the trading system
    trading_system = MLTradingSystem()
    
    try:
        trading_system.start()
    except Exception as e:
        logger.error(f"Failed to start trading system: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()