"""
Trading Strategy Implementation with Buy/Sell Signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import (STOP_LOSS_PERCENTAGE, TAKE_PROFIT_PERCENTAGE, 
                   MAX_POSITION_SIZE, VOLUME_MULTIPLIER)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self):
        self.stop_loss_pct = STOP_LOSS_PERCENTAGE
        self.take_profit_pct = TAKE_PROFIT_PERCENTAGE
        self.max_position_size = MAX_POSITION_SIZE
        self.volume_multiplier = VOLUME_MULTIPLIER
        self.active_positions = {}
        self.trade_history = []
    
    def generate_signals(self, symbol, data, support_levels, resistance_levels, ml_predictions=None):
        """
        Generate comprehensive trading signals
        """
        try:
            signals = []
            current_price = data['Close'].iloc[-1]
            current_time = data.index[-1] if hasattr(data.index, 'name') else datetime.now()
            
            # Technical analysis signals
            tech_signals = self._technical_analysis_signals(data)
            
            # Support/Resistance signals
            sr_signals = self._support_resistance_signals(data, support_levels, resistance_levels)
            
            # Volume analysis signals
            volume_signals = self._volume_analysis_signals(data)
            
            # ML predictions signals
            ml_signals = self._ml_prediction_signals(ml_predictions) if ml_predictions else []
            
            # Combine all signals
            all_signals = tech_signals + sr_signals + volume_signals + ml_signals
            
            # Generate final recommendation
            final_signal = self._combine_signals(all_signals, symbol, current_price, current_time)
            
            return final_signal
            
        except Exception as e:
            logger.error(f"Error generating signals for {symbol}: {str(e)}")
            return None
    
    def _technical_analysis_signals(self, data):
        """
        Generate signals based on technical indicators
        """
        signals = []
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest
        
        # MACD signals
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append({'type': 'buy', 'indicator': 'MACD_bullish_crossover', 'strength': 0.7})
        elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append({'type': 'sell', 'indicator': 'MACD_bearish_crossover', 'strength': 0.7})
        
        # RSI signals
        if latest['RSI'] < 30:
            signals.append({'type': 'buy', 'indicator': 'RSI_oversold', 'strength': 0.6})
        elif latest['RSI'] > 70:
            signals.append({'type': 'sell', 'indicator': 'RSI_overbought', 'strength': 0.6})
        
        # Moving Average signals
        if (latest['Close'] > latest['SMA_20'] and prev['Close'] <= prev['SMA_20']):
            signals.append({'type': 'buy', 'indicator': 'SMA20_breakout', 'strength': 0.5})
        elif (latest['Close'] < latest['SMA_20'] and prev['Close'] >= prev['SMA_20']):
            signals.append({'type': 'sell', 'indicator': 'SMA20_breakdown', 'strength': 0.5})
        
        # Bollinger Bands signals
        bb_position = (latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower'])
        if bb_position < 0.1:  # Near lower band
            signals.append({'type': 'buy', 'indicator': 'BB_oversold', 'strength': 0.4})
        elif bb_position > 0.9:  # Near upper band
            signals.append({'type': 'sell', 'indicator': 'BB_overbought', 'strength': 0.4})
        
        # Golden Cross / Death Cross
        if latest['SMA_10'] > latest['SMA_50'] and prev['SMA_10'] <= prev['SMA_50']:
            signals.append({'type': 'buy', 'indicator': 'golden_cross', 'strength': 0.8})
        elif latest['SMA_10'] < latest['SMA_50'] and prev['SMA_10'] >= prev['SMA_50']:
            signals.append({'type': 'sell', 'indicator': 'death_cross', 'strength': 0.8})
        
        return signals
    
    def _support_resistance_signals(self, data, support_levels, resistance_levels):
        """
        Generate signals based on support/resistance levels
        """
        signals = []
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        
        # Support bounce signals
        for support in support_levels:
            distance_pct = abs(current_price - support) / support
            if distance_pct < 0.01:  # Within 1% of support
                if current_price > prev_price:  # Bouncing up
                    signals.append({
                        'type': 'buy', 
                        'indicator': 'support_bounce', 
                        'strength': 0.8,
                        'level': support,
                        'distance': distance_pct
                    })
        
        # Resistance rejection signals
        for resistance in resistance_levels:
            distance_pct = abs(current_price - resistance) / resistance
            if distance_pct < 0.01:  # Within 1% of resistance
                if current_price < prev_price:  # Rejecting down
                    signals.append({
                        'type': 'sell', 
                        'indicator': 'resistance_rejection', 
                        'strength': 0.8,
                        'level': resistance,
                        'distance': distance_pct
                    })
        
        # Breakout signals
        for resistance in resistance_levels:
            if prev_price < resistance and current_price > resistance:
                breakout_strength = (current_price - resistance) / resistance
                if breakout_strength > 0.02:  # Strong breakout
                    signals.append({
                        'type': 'buy', 
                        'indicator': 'resistance_breakout', 
                        'strength': min(0.9, 0.5 + breakout_strength * 10),
                        'level': resistance
                    })
        
        # Breakdown signals
        for support in support_levels:
            if prev_price > support and current_price < support:
                breakdown_strength = (support - current_price) / support
                if breakdown_strength > 0.02:  # Strong breakdown
                    signals.append({
                        'type': 'sell', 
                        'indicator': 'support_breakdown', 
                        'strength': min(0.9, 0.5 + breakdown_strength * 10),
                        'level': support
                    })
        
        return signals
    
    def _volume_analysis_signals(self, data):
        """
        Generate signals based on volume analysis
        """
        signals = []
        latest = data.iloc[-1]
        
        # Volume confirmation for price movements
        if latest['Volume_Ratio'] > self.volume_multiplier:
            price_change = latest['Price_Change']
            if price_change > 0.02:  # Positive price movement with high volume
                signals.append({
                    'type': 'buy', 
                    'indicator': 'volume_breakout', 
                    'strength': min(0.8, 0.4 + latest['Volume_Ratio'] * 0.1)
                })
            elif price_change < -0.02:  # Negative price movement with high volume
                signals.append({
                    'type': 'sell', 
                    'indicator': 'volume_breakdown', 
                    'strength': min(0.8, 0.4 + latest['Volume_Ratio'] * 0.1)
                })
        
        return signals
    
    def _ml_prediction_signals(self, ml_predictions):
        """
        Convert ML predictions to trading signals
        """
        signals = []
        
        for prediction in ml_predictions:
            if prediction['signal'] == 'buy':
                signals.append({
                    'type': 'buy',
                    'indicator': 'ml_prediction',
                    'strength': prediction['confidence'],
                    'confidence': prediction['confidence']
                })
            elif prediction['signal'] == 'sell':
                signals.append({
                    'type': 'sell',
                    'indicator': 'ml_prediction',
                    'strength': prediction['confidence'],
                    'confidence': prediction['confidence']
                })
        
        return signals
    
    def _combine_signals(self, all_signals, symbol, current_price, current_time):
        """
        Combine all signals into final trading recommendation
        """
        if not all_signals:
            return self._create_signal_output(symbol, 'hold', 0, current_price, current_time, [], "No clear signals")
        
        # Separate buy and sell signals
        buy_signals = [s for s in all_signals if s['type'] == 'buy']
        sell_signals = [s for s in all_signals if s['type'] == 'sell']
        
        # Calculate weighted scores
        buy_score = sum(s['strength'] for s in buy_signals)
        sell_score = sum(s['strength'] for s in sell_signals)
        
        # Determine final action
        if buy_score > sell_score and buy_score > 1.0:
            action = 'buy'
            confidence = min(0.95, buy_score / (buy_score + sell_score))
            reasoning = self._create_reasoning(buy_signals, 'buy')
        elif sell_score > buy_score and sell_score > 1.0:
            action = 'sell'
            confidence = min(0.95, sell_score / (buy_score + sell_score))
            reasoning = self._create_reasoning(sell_signals, 'sell')
        else:
            action = 'hold'
            confidence = 0.5
            reasoning = "Mixed signals or insufficient strength"
        
        return self._create_signal_output(symbol, action, confidence, current_price, current_time, all_signals, reasoning)
    
    def _create_reasoning(self, signals, action):
        """
        Create human-readable reasoning for the trading decision
        """
        if not signals:
            return f"No {action} signals detected"
        
        reasoning_parts = []
        
        # Group signals by indicator type
        signal_groups = {}
        for signal in signals:
            indicator = signal['indicator']
            if indicator not in signal_groups:
                signal_groups[indicator] = []
            signal_groups[indicator].append(signal)
        
        # Create reasoning for each group
        for indicator, group_signals in signal_groups.items():
            if len(group_signals) == 1:
                strength = group_signals[0]['strength']
                reasoning_parts.append(f"{indicator.replace('_', ' ').title()} (strength: {strength:.2f})")
            else:
                avg_strength = np.mean([s['strength'] for s in group_signals])
                reasoning_parts.append(f"{indicator.replace('_', ' ').title()} (avg strength: {avg_strength:.2f})")
        
        return f"{action.title()} signals: " + ", ".join(reasoning_parts)
    
    def _create_signal_output(self, symbol, action, confidence, current_price, timestamp, signals, reasoning):
        """
        Create standardized signal output
        """
        # Calculate risk management levels
        if action == 'buy':
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            risk_reward_ratio = self.take_profit_pct / self.stop_loss_pct
        elif action == 'sell':
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)
            risk_reward_ratio = self.take_profit_pct / self.stop_loss_pct
        else:
            stop_loss = None
            take_profit = None
            risk_reward_ratio = None
        
        return {
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'current_price': current_price,
            'timestamp': timestamp,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'position_size': self.max_position_size if action != 'hold' else 0,
            'reasoning': reasoning,
            'signals': signals,
            'signal_count': len(signals)
        }
    
    def manage_position(self, symbol, current_price, entry_price, action):
        """
        Manage existing positions with stop loss and take profit
        """
        if symbol not in self.active_positions:
            return None
        
        position = self.active_positions[symbol]
        
        if action == 'buy':  # Long position
            # Check stop loss
            if current_price <= position['stop_loss']:
                return {
                    'action': 'sell',
                    'reason': 'stop_loss_triggered',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': (current_price - entry_price) / entry_price
                }
            
            # Check take profit
            if current_price >= position['take_profit']:
                return {
                    'action': 'sell',
                    'reason': 'take_profit_triggered',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': (current_price - entry_price) / entry_price
                }
        
        elif action == 'sell':  # Short position
            # Check stop loss
            if current_price >= position['stop_loss']:
                return {
                    'action': 'buy',
                    'reason': 'stop_loss_triggered',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': (entry_price - current_price) / entry_price
                }
            
            # Check take profit
            if current_price <= position['take_profit']:
                return {
                    'action': 'buy',
                    'reason': 'take_profit_triggered',
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'pnl_pct': (entry_price - current_price) / entry_price
                }
        
        return None
    
    def add_position(self, symbol, action, entry_price, stop_loss, take_profit):
        """
        Add a new position to track
        """
        self.active_positions[symbol] = {
            'action': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': datetime.now()
        }
    
    def close_position(self, symbol, exit_price, reason):
        """
        Close an existing position
        """
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            
            # Calculate P&L
            if position['action'] == 'buy':
                pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
            
            # Add to trade history
            trade = {
                'symbol': symbol,
                'action': position['action'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'entry_time': position['timestamp'],
                'exit_time': datetime.now(),
                'pnl_pct': pnl_pct,
                'reason': reason
            }
            
            self.trade_history.append(trade)
            
            # Remove from active positions
            del self.active_positions[symbol]
            
            return trade
        
        return None
    
    def get_performance_summary(self):
        """
        Get trading performance summary
        """
        if not self.trade_history:
            return None
        
        trades = pd.DataFrame(self.trade_history)
        
        total_trades = len(trades)
        winning_trades = len(trades[trades['pnl_pct'] > 0])
        losing_trades = len(trades[trades['pnl_pct'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = trades[trades['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades[trades['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        total_return = trades['pnl_pct'].sum()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'total_return': total_return,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        }

if __name__ == "__main__":
    # Test the trading strategy
    from data_fetcher import StockDataFetcher
    from support_resistance import SupportResistanceDetector
    from ml_model import StockMLModel
    
    # Initialize components
    fetcher = StockDataFetcher()
    detector = SupportResistanceDetector()
    strategy = TradingStrategy()
    
    # Get test data
    data = fetcher.fetch_live_data("AAPL")
    
    if data is not None:
        # Find support/resistance levels
        levels = detector.find_support_resistance_levels(data)
        
        # Generate signals
        signal = strategy.generate_signals(
            "AAPL", 
            data, 
            levels['support_levels'], 
            levels['resistance_levels']
        )
        
        if signal:
            print(f"Trading Signal for AAPL:")
            print(f"Action: {signal['action']}")
            print(f"Confidence: {signal['confidence']:.2f}")
            print(f"Current Price: ${signal['current_price']:.2f}")
            print(f"Reasoning: {signal['reasoning']}")
            
            if signal['action'] != 'hold':
                print(f"Stop Loss: ${signal['stop_loss']:.2f}")
                print(f"Take Profit: ${signal['take_profit']:.2f}")
                print(f"Risk/Reward Ratio: {signal['risk_reward_ratio']:.2f}")
    else:
        print("Could not fetch data for testing")