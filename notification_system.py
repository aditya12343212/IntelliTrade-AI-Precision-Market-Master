"""
Notification System for Trading Alerts
"""

import requests
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import os
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationSystem:
    def __init__(self):
        self.telegram_token = TELEGRAM_BOT_TOKEN
        self.telegram_chat_id = TELEGRAM_CHAT_ID
        self.notification_history = []
        
    def send_telegram_alert(self, message):
        """
        Send alert via Telegram bot
        """
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured")
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Telegram alert sent successfully")
                return True
            else:
                logger.error(f"Failed to send Telegram alert: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {str(e)}")
            return False
    
    def send_email_alert(self, subject, message, email_config):
        """
        Send alert via email
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = email_config['sender_email']
            msg['To'] = email_config['receiver_email']
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            # Gmail SMTP configuration
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(email_config['sender_email'], email_config['app_password'])
            
            text = msg.as_string()
            server.sendmail(email_config['sender_email'], email_config['receiver_email'], text)
            server.quit()
            
            logger.info("Email alert sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")
            return False
    
    def send_trading_signal_alert(self, signal_data):
        """
        Send formatted trading signal alert
        """
        try:
            # Format the alert message
            message = self._format_trading_signal(signal_data)
            
            # Send via Telegram
            telegram_sent = self.send_telegram_alert(message)
            
            # Log the notification
            self._log_notification(signal_data, message, telegram_sent)
            
            return telegram_sent
            
        except Exception as e:
            logger.error(f"Error sending trading signal alert: {str(e)}")
            return False
    
    def send_breakout_alert(self, symbol, breakout_data):
        """
        Send alert for support/resistance breakouts
        """
        try:
            breakout_type = breakout_data['type']
            level = breakout_data['level']
            current_price = breakout_data['current_price']
            strength = breakout_data['strength']
            
            if 'breakout' in breakout_type:
                emoji = "🚀"
                action = "BUY"
                level_type = "resistance"
            else:
                emoji = "📉"
                action = "SELL"
                level_type = "support"
            
            message = f"""
{emoji} <b>BREAKOUT ALERT</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Type:</b> {breakout_type.replace('_', ' ').title()}
<b>Action:</b> {action}

<b>Current Price:</b> ${current_price:.2f}
<b>{level_type.title()} Level:</b> ${level:.2f}
<b>Strength:</b> {strength:.1%}

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ <i>This is an automated alert. Please do your own analysis before trading.</i>
"""
            
            return self.send_telegram_alert(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending breakout alert: {str(e)}")
            return False
    
    def send_ml_prediction_alert(self, symbol, prediction_data):
        """
        Send alert for ML model predictions
        """
        try:
            signal = prediction_data['signal']
            confidence = prediction_data['confidence']
            
            if signal == 'buy':
                emoji = "📈"
                color = "green"
            elif signal == 'sell':
                emoji = "📉" 
                color = "red"
            else:
                emoji = "⏸️"
                color = "yellow"
            
            message = f"""
{emoji} <b>ML PREDICTION ALERT</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Signal:</b> {signal.upper()}
<b>Confidence:</b> {confidence:.1%}

<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🤖 <i>ML model prediction - Consider combining with technical analysis.</i>
"""
            
            return self.send_telegram_alert(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending ML prediction alert: {str(e)}")
            return False
    
    def send_portfolio_update(self, portfolio_data):
        """
        Send daily portfolio performance update
        """
        try:
            total_value = portfolio_data.get('total_value', 0)
            daily_pnl = portfolio_data.get('daily_pnl', 0)
            daily_pnl_pct = portfolio_data.get('daily_pnl_pct', 0)
            active_positions = portfolio_data.get('active_positions', 0)
            
            pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
            
            message = f"""
📊 <b>DAILY PORTFOLIO UPDATE</b> 📊

<b>Total Portfolio Value:</b> ${total_value:,.2f}
<b>Daily P&L:</b> {pnl_emoji} ${daily_pnl:,.2f} ({daily_pnl_pct:+.2%})
<b>Active Positions:</b> {active_positions}

<b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}

💼 <i>Portfolio summary for today's trading session.</i>
"""
            
            return self.send_telegram_alert(message.strip())
            
        except Exception as e:
            logger.error(f"Error sending portfolio update: {str(e)}")
            return False
    
    def _format_trading_signal(self, signal_data):
        """
        Format trading signal data into readable message
        """
        symbol = signal_data['symbol']
        action = signal_data['action'].upper()
        confidence = signal_data['confidence']
        current_price = signal_data['current_price']
        reasoning = signal_data['reasoning']
        
        # Choose emoji based on action
        if action == 'BUY':
            emoji = "🟢"
        elif action == 'SELL':
            emoji = "🔴"
        else:
            emoji = "🟡"
        
        message = f"""
{emoji} <b>TRADING SIGNAL</b> {emoji}

<b>Symbol:</b> {symbol}
<b>Action:</b> {action}
<b>Confidence:</b> {confidence:.1%}
<b>Current Price:</b> ${current_price:.2f}
"""
        
        # Add risk management info if available
        if action != 'HOLD':
            if signal_data.get('stop_loss'):
                message += f"\n<b>Stop Loss:</b> ${signal_data['stop_loss']:.2f}"
            if signal_data.get('take_profit'):
                message += f"\n<b>Take Profit:</b> ${signal_data['take_profit']:.2f}"
            if signal_data.get('risk_reward_ratio'):
                message += f"\n<b>Risk/Reward:</b> 1:{signal_data['risk_reward_ratio']:.2f}"
        
        message += f"""

<b>Reasoning:</b> {reasoning}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ <i>This is an automated signal. Please verify before trading.</i>
"""
        
        return message.strip()
    
    def _log_notification(self, signal_data, message, sent_successfully):
        """
        Log notification history
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'symbol': signal_data.get('symbol', 'Unknown'),
            'action': signal_data.get('action', 'Unknown'),
            'confidence': signal_data.get('confidence', 0),
            'message': message,
            'sent_successfully': sent_successfully
        }
        
        self.notification_history.append(log_entry)
        
        # Keep only last 100 notifications
        if len(self.notification_history) > 100:
            self.notification_history = self.notification_history[-100:]
    
    def save_notification_history(self, filepath='notification_history.json'):
        """
        Save notification history to file
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.notification_history, f, indent=2, default=str)
            logger.info(f"Notification history saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving notification history: {str(e)}")
    
    def load_notification_history(self, filepath='notification_history.json'):
        """
        Load notification history from file
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    self.notification_history = json.load(f)
                logger.info(f"Notification history loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading notification history: {str(e)}")
    
    def get_notification_stats(self):
        """
        Get statistics about sent notifications
        """
        if not self.notification_history:
            return None
        
        total_notifications = len(self.notification_history)
        successful_notifications = sum(1 for n in self.notification_history if n['sent_successfully'])
        
        # Count by action type
        action_counts = {}
        for notification in self.notification_history:
            action = notification.get('action', 'Unknown')
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'total_notifications': total_notifications,
            'successful_notifications': successful_notifications,
            'success_rate': successful_notifications / total_notifications if total_notifications > 0 else 0,
            'action_breakdown': action_counts
        }
    
    def test_notification_system(self):
        """
        Test the notification system
        """
        test_message = f"""
🧪 <b>NOTIFICATION SYSTEM TEST</b> 🧪

This is a test message from your ML Stock Trading System.

<b>System Status:</b> ✅ Active
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

If you receive this message, your notification system is working correctly!
"""
        
        result = self.send_telegram_alert(test_message.strip())
        
        if result:
            logger.info("Notification system test successful")
        else:
            logger.error("Notification system test failed")
        
        return result

if __name__ == "__main__":
    # Test the notification system
    notifier = NotificationSystem()
    
    # Test basic functionality
    test_result = notifier.test_notification_system()
    print(f"Notification test result: {test_result}")
    
    # Test trading signal alert
    sample_signal = {
        'symbol': 'AAPL',
        'action': 'buy',
        'confidence': 0.85,
        'current_price': 150.25,
        'stop_loss': 142.75,
        'take_profit': 165.50,
        'risk_reward_ratio': 2.1,
        'reasoning': 'Strong technical breakout with volume confirmation'
    }
    
    signal_result = notifier.send_trading_signal_alert(sample_signal)
    print(f"Trading signal alert result: {signal_result}")
    
    # Show notification stats
    stats = notifier.get_notification_stats()
    if stats:
        print("Notification statistics:", stats)