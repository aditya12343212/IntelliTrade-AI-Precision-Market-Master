"""
Support and Resistance Level Detection Algorithm
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import logging
from config import SUPPORT_RESISTANCE_WINDOW, MIN_TOUCHES, BREAKOUT_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupportResistanceDetector:
    def __init__(self, window=SUPPORT_RESISTANCE_WINDOW, min_touches=MIN_TOUCHES):
        self.window = window
        self.min_touches = min_touches
        self.breakout_threshold = BREAKOUT_THRESHOLD
    
    def find_support_resistance_levels(self, data):
        """
        Find support and resistance levels using multiple methods
        """
        try:
            support_levels = []
            resistance_levels = []
            
            # Method 1: Local extrema
            local_support, local_resistance = self._find_local_extrema(data)
            support_levels.extend(local_support)
            resistance_levels.extend(local_resistance)
            
            # Method 2: Volume weighted levels
            volume_support, volume_resistance = self._find_volume_weighted_levels(data)
            support_levels.extend(volume_support)
            resistance_levels.extend(volume_resistance)
            
            # Method 3: Psychological levels (round numbers)
            psych_support, psych_resistance = self._find_psychological_levels(data)
            support_levels.extend(psych_support)
            resistance_levels.extend(psych_resistance)
            
            # Consolidate and validate levels
            support_levels = self._consolidate_levels(support_levels, data, 'support')
            resistance_levels = self._consolidate_levels(resistance_levels, data, 'resistance')
            
            return {
                'support_levels': sorted(support_levels),
                'resistance_levels': sorted(resistance_levels, reverse=True),
                'current_price': data['Close'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance levels: {str(e)}")
            return {'support_levels': [], 'resistance_levels': [], 'current_price': None}
    
    def _find_local_extrema(self, data):
        """
        Find local minima (support) and maxima (resistance) using scipy
        """
        high_prices = data['High'].values
        low_prices = data['Low'].values
        
        # Find local minima (support levels)
        local_minima_idx = argrelextrema(low_prices, np.less, order=self.window//2)[0]
        support_levels = low_prices[local_minima_idx].tolist()
        
        # Find local maxima (resistance levels)
        local_maxima_idx = argrelextrema(high_prices, np.greater, order=self.window//2)[0]
        resistance_levels = high_prices[local_maxima_idx].tolist()
        
        return support_levels, resistance_levels
    
    def _find_volume_weighted_levels(self, data):
        """
        Find levels with high volume activity
        """
        # Create price bins and calculate volume at each level
        price_min = data['Low'].min()
        price_max = data['High'].max()
        price_range = price_max - price_min
        bin_size = price_range / 100  # 100 bins
        
        volume_profile = {}
        
        for _, row in data.iterrows():
            # Distribute volume across the high-low range
            price_levels = np.arange(row['Low'], row['High'], bin_size)
            volume_per_level = row['Volume'] / len(price_levels) if len(price_levels) > 0 else row['Volume']
            
            for price in price_levels:
                bin_price = round(price / bin_size) * bin_size
                volume_profile[bin_price] = volume_profile.get(bin_price, 0) + volume_per_level
        
        # Find high volume levels
        if not volume_profile:
            return [], []
        
        avg_volume = np.mean(list(volume_profile.values()))
        high_volume_levels = [price for price, volume in volume_profile.items() 
                            if volume > avg_volume * 1.5]
        
        current_price = data['Close'].iloc[-1]
        support_levels = [level for level in high_volume_levels if level < current_price]
        resistance_levels = [level for level in high_volume_levels if level > current_price]
        
        return support_levels, resistance_levels
    
    def _find_psychological_levels(self, data):
        """
        Find psychological support/resistance levels (round numbers)
        """
        current_price = data['Close'].iloc[-1]
        price_min = data['Low'].min()
        price_max = data['High'].max()
        
        # Generate round number levels
        psychological_levels = []
        
        # For prices under $10, use $0.50 intervals
        if current_price < 10:
            step = 0.5
        # For prices under $100, use $1 intervals
        elif current_price < 100:
            step = 1
        # For prices under $1000, use $5 intervals
        elif current_price < 1000:
            step = 5
        # For higher prices, use $10 intervals
        else:
            step = 10
        
        # Generate levels around the current price range
        start_level = int(price_min / step) * step
        end_level = int(price_max / step + 1) * step
        
        level = start_level
        while level <= end_level:
            psychological_levels.append(level)
            level += step
        
        # Separate into support and resistance
        support_levels = [level for level in psychological_levels if level < current_price]
        resistance_levels = [level for level in psychological_levels if level > current_price]
        
        return support_levels, resistance_levels
    
    def _consolidate_levels(self, levels, data, level_type):
        """
        Consolidate similar levels and validate with historical touches
        """
        if not levels:
            return []
        
        # Remove duplicates and sort
        levels = sorted(list(set(levels)))
        
        # Group similar levels (within 1% of each other)
        consolidated = []
        current_group = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_group[-1]) / current_group[-1] < 0.01:  # Within 1%
                current_group.append(level)
            else:
                # Take average of the group
                consolidated.append(np.mean(current_group))
                current_group = [level]
        
        # Don't forget the last group
        consolidated.append(np.mean(current_group))
        
        # Validate levels with historical touches
        validated_levels = []
        for level in consolidated:
            touches = self._count_touches(data, level, level_type)
            if touches >= self.min_touches:
                validated_levels.append(level)
        
        return validated_levels
    
    def _count_touches(self, data, level, level_type, tolerance=0.02):
        """
        Count how many times price has touched a support/resistance level
        """
        touches = 0
        
        if level_type == 'support':
            # Count touches at low prices
            for low in data['Low']:
                if abs(low - level) / level <= tolerance:
                    touches += 1
        else:  # resistance
            # Count touches at high prices
            for high in data['High']:
                if abs(high - level) / level <= tolerance:
                    touches += 1
        
        return touches
    
    def detect_breakouts(self, data, support_levels, resistance_levels):
        """
        Detect breakouts from support/resistance levels
        """
        breakouts = []
        current_price = data['Close'].iloc[-1]
        previous_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        
        # Check for support breakdowns
        for support in support_levels:
            if (previous_price > support and current_price < support and
                (support - current_price) / support > self.breakout_threshold):
                breakouts.append({
                    'type': 'support_breakdown',
                    'level': support,
                    'current_price': current_price,
                    'strength': (support - current_price) / support
                })
        
        # Check for resistance breakouts
        for resistance in resistance_levels:
            if (previous_price < resistance and current_price > resistance and
                (current_price - resistance) / resistance > self.breakout_threshold):
                breakouts.append({
                    'type': 'resistance_breakout',
                    'level': resistance,
                    'current_price': current_price,
                    'strength': (current_price - resistance) / resistance
                })
        
        return breakouts
    
    def get_nearest_levels(self, data, support_levels, resistance_levels):
        """
        Get the nearest support and resistance levels to current price
        """
        current_price = data['Close'].iloc[-1]
        
        # Find nearest support (below current price)
        valid_support = [s for s in support_levels if s < current_price]
        nearest_support = max(valid_support) if valid_support else None
        
        # Find nearest resistance (above current price)
        valid_resistance = [r for r in resistance_levels if r > current_price]
        nearest_resistance = min(valid_resistance) if valid_resistance else None
        
        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance': ((current_price - nearest_support) / current_price * 100) if nearest_support else None,
            'resistance_distance': ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None
        }

if __name__ == "__main__":
    # Test the detector
    from data_fetcher import StockDataFetcher
    
    fetcher = StockDataFetcher()
    detector = SupportResistanceDetector()
    
    # Get test data
    data = fetcher.fetch_live_data("AAPL")
    
    if data is not None:
        # Find support/resistance levels
        levels = detector.find_support_resistance_levels(data)
        print("Support levels:", levels['support_levels'])
        print("Resistance levels:", levels['resistance_levels'])
        print("Current price:", levels['current_price'])
        
        # Check for breakouts
        breakouts = detector.detect_breakouts(data, levels['support_levels'], levels['resistance_levels'])
        if breakouts:
            print("Breakouts detected:", breakouts)
        
        # Get nearest levels
        nearest = detector.get_nearest_levels(data, levels['support_levels'], levels['resistance_levels'])
        print("Nearest levels:", nearest)