"""
Machine Learning Model for Stock Pattern Recognition and Prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import PREDICTION_CONFIDENCE_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockMLModel:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.confidence_threshold = PREDICTION_CONFIDENCE_THRESHOLD
        self.best_model = None
        self.is_trained = False
    
    def prepare_features(self, data, support_levels=None, resistance_levels=None):
        """
        Prepare features for machine learning model
        """
        try:
            features = data.copy()
            
            # Price-based features
            features['price_to_sma10'] = features['Close'] / features['SMA_10']
            features['price_to_sma20'] = features['Close'] / features['SMA_20']
            features['price_to_sma50'] = features['Close'] / features['SMA_50']
            
            # Volatility features
            features['volatility'] = features['High_Low_Ratio']
            features['price_volatility'] = features['Price_Change'].rolling(window=10).std()
            
            # Momentum features
            features['momentum_5'] = features['Close'] / features['Close'].shift(5) - 1
            features['momentum_10'] = features['Close'] / features['Close'].shift(10) - 1
            features['momentum_20'] = features['Close'] / features['Close'].shift(20) - 1
            
            # MACD features
            features['macd_signal'] = np.where(features['MACD'] > features['MACD_Signal'], 1, 0)
            features['macd_histogram_trend'] = features['MACD_Histogram'].diff()
            
            # RSI features
            features['rsi_oversold'] = np.where(features['RSI'] < 30, 1, 0)
            features['rsi_overbought'] = np.where(features['RSI'] > 70, 1, 0)
            features['rsi_trend'] = features['RSI'].diff()
            
            # Bollinger Bands features
            features['bb_position'] = (features['Close'] - features['BB_Lower']) / (features['BB_Upper'] - features['BB_Lower'])
            features['bb_squeeze'] = (features['BB_Upper'] - features['BB_Lower']) / features['BB_Middle']
            
            # Volume features
            features['volume_trend'] = features['Volume'].diff()
            features['volume_spike'] = np.where(features['Volume_Ratio'] > 2, 1, 0)
            
            # Support/Resistance features
            if support_levels and resistance_levels:
                features['distance_to_support'] = self._calculate_distance_to_levels(features['Close'], support_levels, 'support')
                features['distance_to_resistance'] = self._calculate_distance_to_levels(features['Close'], resistance_levels, 'resistance')
                features['near_support'] = np.where(features['distance_to_support'] < 0.02, 1, 0)
                features['near_resistance'] = np.where(features['distance_to_resistance'] < 0.02, 1, 0)
            
            # Pattern recognition features
            features['doji'] = self._detect_doji(features)
            features['hammer'] = self._detect_hammer(features)
            features['engulfing'] = self._detect_engulfing(features)
            
            # Trend features
            features['trend_short'] = self._calculate_trend(features['Close'], 5)
            features['trend_medium'] = self._calculate_trend(features['Close'], 10)
            features['trend_long'] = self._calculate_trend(features['Close'], 20)
            
            # Time-based features
            features['hour'] = features.index.hour if hasattr(features.index, 'hour') else 0
            features['day_of_week'] = features.index.dayofweek if hasattr(features.index, 'dayofweek') else 0
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return data
    
    def _calculate_distance_to_levels(self, prices, levels, level_type):
        """
        Calculate distance to nearest support/resistance level
        """
        distances = []
        for price in prices:
            if level_type == 'support':
                valid_levels = [l for l in levels if l < price]
                nearest = max(valid_levels) if valid_levels else price * 0.95
            else:  # resistance
                valid_levels = [l for l in levels if l > price]
                nearest = min(valid_levels) if valid_levels else price * 1.05
            
            distance = abs(price - nearest) / price
            distances.append(distance)
        
        return distances
    
    def _detect_doji(self, data):
        """
        Detect Doji candlestick pattern
        """
        body_size = abs(data['Close'] - data['Open'])
        candle_range = data['High'] - data['Low']
        return np.where((body_size / candle_range < 0.1) & (candle_range > 0), 1, 0)
    
    def _detect_hammer(self, data):
        """
        Detect Hammer candlestick pattern
        """
        body_size = abs(data['Close'] - data['Open'])
        lower_shadow = data[['Close', 'Open']].min(axis=1) - data['Low']
        upper_shadow = data['High'] - data[['Close', 'Open']].max(axis=1)
        
        return np.where(
            (lower_shadow > 2 * body_size) & 
            (upper_shadow < 0.1 * body_size) & 
            (body_size > 0), 1, 0
        )
    
    def _detect_engulfing(self, data):
        """
        Detect Engulfing candlestick pattern
        """
        prev_body = abs(data['Close'].shift(1) - data['Open'].shift(1))
        curr_body = abs(data['Close'] - data['Open'])
        
        bullish_engulfing = (
            (data['Close'].shift(1) < data['Open'].shift(1)) &  # Previous red candle
            (data['Close'] > data['Open']) &  # Current green candle
            (data['Open'] < data['Close'].shift(1)) &  # Current opens below previous close
            (data['Close'] > data['Open'].shift(1)) &  # Current closes above previous open
            (curr_body > prev_body)  # Current body larger than previous
        )
        
        return np.where(bullish_engulfing, 1, 0)
    
    def _calculate_trend(self, prices, window):
        """
        Calculate trend direction over a window
        """
        trend = []
        for i in range(len(prices)):
            if i < window:
                trend.append(0)
            else:
                slope = np.polyfit(range(window), prices.iloc[i-window+1:i+1], 1)[0]
                if slope > 0.01:
                    trend.append(1)  # Uptrend
                elif slope < -0.01:
                    trend.append(-1)  # Downtrend
                else:
                    trend.append(0)  # Sideways
        
        return trend
    
    def create_labels(self, data, future_periods=5):
        """
        Create labels for prediction (future price movement)
        """
        labels = []
        
        for i in range(len(data)):
            if i >= len(data) - future_periods:
                labels.append('hold')  # Not enough future data
                continue
            
            current_price = data['Close'].iloc[i]
            future_high = data['High'].iloc[i+1:i+future_periods+1].max()
            future_low = data['Low'].iloc[i+1:i+future_periods+1].min()
            
            upside_potential = (future_high - current_price) / current_price
            downside_risk = (current_price - future_low) / current_price
            
            # Define trading signals based on risk/reward
            if upside_potential > 0.03 and upside_potential > downside_risk * 1.5:
                labels.append('buy')
            elif downside_risk > 0.03 and downside_risk > upside_potential * 1.5:
                labels.append('sell')
            else:
                labels.append('hold')
        
        return labels
    
    def train_model(self, features_data, labels):
        """
        Train the machine learning models
        """
        try:
            # Prepare feature matrix
            feature_columns = [col for col in features_data.columns 
                             if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            X = features_data[feature_columns].fillna(0)
            y = labels
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train models and find the best one
            best_score = 0
            best_model_name = None
            
            for name, model in self.models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                avg_score = cv_scores.mean()
                
                logger.info(f"{name} CV Score: {avg_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model_name = name
            
            # Set best model
            self.best_model = self.models[best_model_name]
            self.feature_columns = feature_columns
            self.is_trained = True
            
            # Evaluate on test set
            y_pred = self.best_model.predict(X_test)
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Test Accuracy: {self.best_model.score(X_test, y_test):.4f}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def predict(self, features_data):
        """
        Make predictions using the trained model
        """
        if not self.is_trained or self.best_model is None:
            logger.error("Model not trained yet")
            return None
        
        try:
            # Prepare features
            X = features_data[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Get prediction and probability
            prediction = self.best_model.predict(X_scaled)
            probabilities = self.best_model.predict_proba(X_scaled)
            
            # Convert back to labels
            predicted_labels = self.label_encoder.inverse_transform(prediction)
            
            # Get confidence (max probability)
            confidence = np.max(probabilities, axis=1)
            
            # Filter by confidence threshold
            results = []
            for i, (label, conf) in enumerate(zip(predicted_labels, confidence)):
                if conf >= self.confidence_threshold:
                    results.append({
                        'signal': label,
                        'confidence': conf,
                        'timestamp': features_data.index[i] if hasattr(features_data, 'index') else i
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def save_model(self, filepath):
        """
        Save the trained model and scaler
        """
        try:
            model_data = {
                'best_model': self.best_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath):
        """
        Load a trained model and scaler
        """
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['best_model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

if __name__ == "__main__":
    # Test the ML model
    from data_fetcher import StockDataFetcher
    from support_resistance import SupportResistanceDetector
    
    # Initialize components
    fetcher = StockDataFetcher()
    detector = SupportResistanceDetector()
    model = StockMLModel()
    
    # Get test data
    data = fetcher.fetch_live_data("AAPL", period="60d")  # More data for training
    
    if data is not None and len(data) > 50:
        # Find support/resistance levels
        levels = detector.find_support_resistance_levels(data)
        
        # Prepare features
        features = model.prepare_features(data, levels['support_levels'], levels['resistance_levels'])
        
        # Create labels
        labels = model.create_labels(features)
        
        # Train model
        if len(features) == len(labels):
            success = model.train_model(features, labels)
            
            if success:
                # Make predictions on recent data
                recent_features = features.tail(10)
                predictions = model.predict(recent_features)
                
                if predictions:
                    print("Recent predictions:")
                    for pred in predictions:
                        print(f"Signal: {pred['signal']}, Confidence: {pred['confidence']:.2f}")
                
                # Save model
                model.save_model("stock_ml_model.pkl")
        else:
            print("Feature and label length mismatch")
    else:
        print("Insufficient data for training")