"""
Feature Engineering Module for Crypto Intelligence System
Transforms raw data into ML-ready features
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureSet:
    """Container for engineered features"""
    sentiment_features: Dict[str, float]
    market_features: Dict[str, float]
    social_features: Dict[str, float]
    technical_features: Dict[str, float]
    timestamp: datetime


class FeatureEngineer:
    """
    Feature engineering for crypto market analysis
    
    Creates features from:
    - Sentiment data (Reddit, News)
    - Market data (price, volume)
    - Social metrics (engagement, velocity)
    - Technical indicators
    """
    
    def __init__(self):
        logger.info("FeatureEngineer initialized")
    
    def engineer_sentiment_features(
        self,
        sentiment_scores: List[float],
        timestamps: List[datetime] = None
    ) -> Dict[str, float]:
        """
        Engineer features from sentiment scores
        
        Args:
            sentiment_scores: List of sentiment scores (-1 to 1)
            timestamps: Optional list of timestamps
            
        Returns:
            Dictionary of engineered features
        """
        if not sentiment_scores:
            return self._empty_sentiment_features()
        
        scores = np.array(sentiment_scores)
        
        features = {
            'sentiment_mean': float(np.mean(scores)),
            'sentiment_std': float(np.std(scores)),
            'sentiment_min': float(np.min(scores)),
            'sentiment_max': float(np.max(scores)),
            'sentiment_range': float(np.max(scores) - np.min(scores)),
            'sentiment_positive_ratio': float(np.sum(scores > 0) / len(scores)),
            'sentiment_negative_ratio': float(np.sum(scores < 0) / len(scores)),
            'sentiment_neutral_ratio': float(np.sum(scores == 0) / len(scores)),
            'sentiment_skewness': float(self._calculate_skewness(scores)),
        }
        
        # Calculate velocity if timestamps provided
        if timestamps and len(timestamps) == len(scores):
            features['sentiment_velocity'] = self._calculate_velocity(scores, timestamps)
        else:
            features['sentiment_velocity'] = 0.0
        
        return features
    
    def engineer_market_features(
        self,
        prices: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Engineer features from OHLCV data
        
        Args:
            prices: DataFrame with open, high, low, close, volume
            
        Returns:
            Dictionary of market features
        """
        if prices.empty:
            return self._empty_market_features()
        
        close = prices['close'].values
        volume = prices['volume'].values
        
        # Price features
        returns = np.diff(close) / close[:-1] if len(close) > 1 else np.array([0])
        
        features = {
            'price_current': float(close[-1]),
            'price_change_1d': float((close[-1] - close[0]) / close[0]) if len(close) > 1 else 0.0,
            'volatility': float(np.std(returns)) if len(returns) > 0 else 0.0,
            'volume_mean': float(np.mean(volume)),
            'volume_current': float(volume[-1]),
            'volume_ratio': float(volume[-1] / np.mean(volume)) if np.mean(volume) > 0 else 1.0,
        }
        
        # Technical indicators if enough data
        if len(close) >= 20:
            features['sma_20'] = float(np.mean(close[-20:]))
            features['price_vs_sma20'] = float((close[-1] - features['sma_20']) / features['sma_20'])
        else:
            features['sma_20'] = float(close[-1])
            features['price_vs_sma20'] = 0.0
        
        # RSI
        features['rsi'] = self._calculate_rsi(close)
        
        return features
    
    def engineer_social_features(
        self,
        posts_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Engineer features from social media data
        
        Args:
            posts_data: DataFrame with post metrics
            
        Returns:
            Dictionary of social features
        """
        if posts_data.empty:
            return self._empty_social_features()
        
        features = {
            'post_count': float(len(posts_data)),
            'avg_score': float(posts_data['score'].mean()) if 'score' in posts_data else 0.0,
            'total_engagement': float(posts_data['score'].sum()) if 'score' in posts_data else 0.0,
            'avg_comments': float(posts_data['num_comments'].mean()) if 'num_comments' in posts_data else 0.0,
        }
        
        # Upvote ratio if available
        if 'upvote_ratio' in posts_data:
            features['avg_upvote_ratio'] = float(posts_data['upvote_ratio'].mean())
        else:
            features['avg_upvote_ratio'] = 0.5
        
        # Post velocity (posts per hour)
        if 'created_utc' in posts_data:
            time_range = (posts_data['created_utc'].max() - posts_data['created_utc'].min())
            hours = time_range.total_seconds() / 3600 if hasattr(time_range, 'total_seconds') else 1
            features['post_velocity'] = features['post_count'] / max(hours, 1)
        else:
            features['post_velocity'] = 0.0
        
        return features
    
    def create_feature_set(
        self,
        sentiment_scores: List[float] = None,
        market_data: pd.DataFrame = None,
        social_data: pd.DataFrame = None
    ) -> FeatureSet:
        """
        Create a complete feature set from all data sources
        
        Args:
            sentiment_scores: List of sentiment scores
            market_data: OHLCV DataFrame
            social_data: Social posts DataFrame
            
        Returns:
            Complete FeatureSet object
        """
        sentiment_features = self.engineer_sentiment_features(sentiment_scores or [])
        market_features = self.engineer_market_features(market_data if market_data is not None else pd.DataFrame())
        social_features = self.engineer_social_features(social_data if social_data is not None else pd.DataFrame())
        
        # Technical features from market data
        technical_features = self._extract_technical_features(market_data)
        
        return FeatureSet(
            sentiment_features=sentiment_features,
            market_features=market_features,
            social_features=social_features,
            technical_features=technical_features,
            timestamp=datetime.utcnow()
        )
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return float(np.mean(((data - mean) / std) ** 3))
    
    def _calculate_velocity(
        self,
        values: np.ndarray,
        timestamps: List[datetime]
    ) -> float:
        """Calculate rate of change"""
        if len(values) < 2:
            return 0.0
        
        time_diff = (timestamps[-1] - timestamps[0]).total_seconds() / 3600  # hours
        if time_diff == 0:
            return 0.0
        
        return float((values[-1] - values[0]) / time_diff)
    
    def _extract_technical_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract additional technical features"""
        if market_data is None or market_data.empty:
            return {'momentum': 0.0, 'trend': 0.0}
        
        close = market_data['close'].values
        
        # Momentum (rate of change)
        if len(close) >= 10:
            momentum = (close[-1] - close[-10]) / close[-10]
        else:
            momentum = 0.0
        
        # Trend (slope of last N candles)
        if len(close) >= 5:
            x = np.arange(5)
            slope, _ = np.polyfit(x, close[-5:], 1)
            trend = slope / close[-5]  # Normalized
        else:
            trend = 0.0
        
        return {
            'momentum': float(momentum),
            'trend': float(trend)
        }
    
    @staticmethod
    def _empty_sentiment_features() -> Dict[str, float]:
        return {
            'sentiment_mean': 0.0,
            'sentiment_std': 0.0,
            'sentiment_min': 0.0,
            'sentiment_max': 0.0,
            'sentiment_range': 0.0,
            'sentiment_positive_ratio': 0.0,
            'sentiment_negative_ratio': 0.0,
            'sentiment_neutral_ratio': 1.0,
            'sentiment_skewness': 0.0,
            'sentiment_velocity': 0.0
        }
    
    @staticmethod
    def _empty_market_features() -> Dict[str, float]:
        return {
            'price_current': 0.0,
            'price_change_1d': 0.0,
            'volatility': 0.0,
            'volume_mean': 0.0,
            'volume_current': 0.0,
            'volume_ratio': 1.0,
            'sma_20': 0.0,
            'price_vs_sma20': 0.0,
            'rsi': 50.0
        }
    
    @staticmethod
    def _empty_social_features() -> Dict[str, float]:
        return {
            'post_count': 0.0,
            'avg_score': 0.0,
            'total_engagement': 0.0,
            'avg_comments': 0.0,
            'avg_upvote_ratio': 0.5,
            'post_velocity': 0.0
        }


# Global instance
_engineer: Optional[FeatureEngineer] = None

def get_feature_engineer() -> FeatureEngineer:
    """Get global feature engineer instance"""
    global _engineer
    if _engineer is None:
        _engineer = FeatureEngineer()
    return _engineer


if __name__ == "__main__":
    # Test feature engineering
    engineer = FeatureEngineer()
    
    # Test sentiment features
    sentiment_scores = [0.5, -0.2, 0.8, 0.1, -0.5, 0.3]
    sentiment_features = engineer.engineer_sentiment_features(sentiment_scores)
    print("Sentiment Features:", sentiment_features)
    
    # Test with mock market data
    market_data = pd.DataFrame({
        'open': [95000, 95500, 96000, 95800, 96200],
        'high': [96000, 96500, 97000, 96500, 97000],
        'low': [94500, 95000, 95500, 95300, 95800],
        'close': [95500, 96000, 95800, 96200, 96800],
        'volume': [1000, 1200, 1100, 1300, 1500]
    })
    market_features = engineer.engineer_market_features(market_data)
    print("\nMarket Features:", market_features)
